/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_dnafold_bond_pre.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "special.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace LAMMPS_NS;
using namespace FixConst;

FixDnafoldBondPre::FixDnafoldBondPre(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), complementarity_file(nullptr)
{
  if (narg != 7) error->all(FLERR,"Illegal fix dnafold/bond/pre command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/bond/pre command");

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  vector_flag = 1;
  size_vector = 4;  // create, remove, total create, total remove
  global_freq = 1;
  extvector = 0;

  // Bonds form between type 1 and type 2 atoms only
  iatomtype = 1;
  jatomtype = 2;

  cutoffsq = utils::numeric(FLERR,arg[4],false,lmp);
  if (cutoffsq <= 0.0) error->all(FLERR,"Illegal cutoff");
  cutoffsq = cutoffsq * cutoffsq;

  dummy_btype = utils::inumeric(FLERR,arg[5],false,lmp);
  if (dummy_btype <= 0) error->all(FLERR,"Illegal dummy bond type");

  complementarity_file = utils::strdup(arg[6]);

  createcount = 0;
  removecount = 0;
  createcounttotal = 0;
  removecounttotal = 0;
  
  // Read complementarity file - only rank 0 reads, then broadcasts
  read_complementarity_file();
}

FixDnafoldBondPre::~FixDnafoldBondPre()
{
  delete[] complementarity_file;
}

int FixDnafoldBondPre::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

void FixDnafoldBondPre::init()
{
  // Find the i_hyb_status property (now an integer)
  int flag_hyb, cols_hyb;
  hyb_status_index = atom->find_custom("hyb_status", flag_hyb, cols_hyb);
  if (hyb_status_index < 0)
    error->all(FLERR,"Could not find i_hyb_status property for fix dnafold/bond/pre");
  if (flag_hyb != 0)
    error->all(FLERR,"Property i_hyb_status must be integer");

  // Find the i_size property (now an integer)
  int flag_size, cols_size;
  size_index = atom->find_custom("size", flag_size, cols_size);
  if (size_index < 0)
    error->all(FLERR,"Could not find i_size property for fix dnafold/bond/pre");
  if (flag_size != 0)
    error->all(FLERR,"Property i_size must be integer");

  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/bond/pre with non-molecular system");
  
  if (force->bond == nullptr)
    error->all(FLERR,"Must define bond_style");

  if (dummy_btype > atom->nbondtypes)
    error->all(FLERR,"Invalid dummy bond type");

  if (force->newton_bond == 0)
    error->all(FLERR,"Fix dnafold/bond/pre requires newton bond on");
}

void FixDnafoldBondPre::init_list(int /* id */, NeighList *ptr)
{
  list = ptr;
}

void FixDnafoldBondPre::setup(int /* vflag */)
{
  post_integrate();
}

void FixDnafoldBondPre::read_complementarity_file()
{
  // Only rank 0 reads the file
  if (me == 0) {
    std::ifstream file(complementarity_file);
    if (!file.is_open()) 
      error->one(FLERR,fmt::format("Cannot open complementarity file '{}'", complementarity_file));
    
    std::string line;
    int line_num = 0;
    enum Section { NONE, TYPES, PAIRS };
    Section current_section = NONE;
    
    while (std::getline(file, line)) {
      line_num++;
      
      // Skip empty lines
      if (line.empty()) continue;
      
      // Skip comment lines (lines starting with #)
      if (line[0] == '#') continue;
      
      // Trim leading whitespace for section detection
      size_t start = line.find_first_not_of(" \t");
      if (start == std::string::npos) continue;  // All whitespace
      std::string trimmed = line.substr(start);
      
      // Check for section headers
      if (trimmed == "TYPES") {
        current_section = TYPES;
        continue;
      } else if (trimmed == "PAIRS") {
        current_section = PAIRS;
        continue;
      }
      
      // Skip TYPES section (bond_pre doesn't need it)
      if (current_section == TYPES) {
        continue;
      }
      
      // Parse PAIRS section
      if (current_section == PAIRS) {
        std::istringstream iss(line);
        tagint tag1, tag2;
        double energy;  // Read but don't store
        
        if (!(iss >> tag1 >> tag2 >> energy)) {
          error->one(FLERR,fmt::format("Invalid PAIRS format in '{}' at line {}", 
                                       complementarity_file, line_num));
        }
        
        // Ensure tag1 < tag2 for consistent lookup
        if (tag1 > tag2) std::swap(tag1, tag2);
        
        // Store in set (we only care if pair exists, not the energy)
        complementarity_set.insert(std::make_pair(tag1, tag2));
        
      } else {
        error->one(FLERR,fmt::format("Line {} in '{}' appears before any section header", 
                                     line_num, complementarity_file));
      }
    }
    
    file.close();
    
    if (complementarity_set.empty()) {
      error->warning(FLERR,"Complementarity file '{}' contains no valid pairs", complementarity_file);
    }
  }
  
  // Broadcast set size
  int set_size = complementarity_set.size();
  MPI_Bcast(&set_size, 1, MPI_INT, 0, world);
  
  // Broadcast set contents
  if (set_size > 0) {
    std::vector<tagint> tags1(set_size);
    std::vector<tagint> tags2(set_size);
    
    if (me == 0) {
      int idx = 0;
      for (const auto &pair : complementarity_set) {
        tags1[idx] = pair.first;
        tags2[idx] = pair.second;
        idx++;
      }
    }
    
    MPI_Bcast(tags1.data(), set_size, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(tags2.data(), set_size, MPI_LMP_TAGINT, 0, world);
    
    if (me != 0) {
      for (int i = 0; i < set_size; i++) {
        complementarity_set.insert(std::make_pair(tags1[i], tags2[i]));
      }
    }
  }
  
  if (me == 0) {
    if (screen) 
      fprintf(screen,"Fix dnafold/bond/pre: Read %d complementarity pairs from '%s'\n",
              set_size, complementarity_file);
    if (logfile) 
      fprintf(logfile,"Fix dnafold/bond/pre: Read %d complementarity pairs from '%s'\n",
              set_size, complementarity_file);
  }
}

bool FixDnafoldBondPre::is_complementary(tagint tag_i, tagint tag_j)
{
  // Ensure tag1 < tag2 for consistent lookup
  tagint tag1 = (tag_i < tag_j) ? tag_i : tag_j;
  tagint tag2 = (tag_i < tag_j) ? tag_j : tag_i;
  
  return complementarity_set.find(std::make_pair(tag1, tag2)) != complementarity_set.end();
}

bool FixDnafoldBondPre::dummy_bond_exists(int i, int j)
{
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  tagint jtag = atom->tag[j];

  // Check if i has a dummy bond to j
  for (int k = 0; k < num_bond[i]; k++) {
    if (bond_atom[i][k] == jtag && bond_type[i][k] == dummy_btype) return true;
  }
  return false;
}

void FixDnafoldBondPre::post_integrate()
{
  int i,j,ii,jj,inum,jnum,k;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int itype, jtype;
  tagint itag, jtag;

  if (update->ntimestep % nevery) return;

  // Acquire updated ghost atom positions
  comm->forward_comm();

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int nlocal = atom->nlocal;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;

  createcount = 0;
  removecount = 0;

  // First pass: remove dummy bonds that are too far apart
  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    
    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    // Check all bonds of atom i
    k = 0;
    while (k < num_bond[i]) {
      // Only check dummy bonds
      if (bond_type[i][k] != dummy_btype) {
        k++;
        continue;
      }
      
      // Find the partner atom
      jtag = bond_atom[i][k];
      j = atom->map(jtag);
      
      if (j < 0) {
        // Partner atom not found - remove bond
        // Shift remaining bonds down
        for (int m = k; m < num_bond[i] - 1; m++) {
          bond_type[i][m] = bond_type[i][m+1];
          bond_atom[i][m] = bond_atom[i][m+1];
        }
        num_bond[i]--;
        removecount++;
        continue;  // Don't increment k since we shifted
      }
      
      // Check distance with minimum image convention
      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      if (domain->xperiodic) {
        if (delx > domain->xprd_half) delx -= domain->xprd;
        else if (delx < -domain->xprd_half) delx += domain->xprd;
      }
      if (domain->yperiodic) {
        if (dely > domain->yprd_half) dely -= domain->yprd;
        else if (dely < -domain->yprd_half) dely += domain->yprd;
      }
      if (domain->zperiodic) {
        if (delz > domain->zprd_half) delz -= domain->zprd;
        else if (delz < -domain->zprd_half) delz += domain->zprd;
      }
      rsq = delx*delx + dely*dely + delz*delz;
      
      if (rsq > cutoffsq) {
        // Too far - remove bond
        // Shift remaining bonds down
        for (int m = k; m < num_bond[i] - 1; m++) {
          bond_type[i][m] = bond_type[i][m+1];
          bond_atom[i][m] = bond_atom[i][m+1];
        }
        num_bond[i]--;
        removecount++;
        continue;  // Don't increment k since we shifted
      }
      
      k++;  // Only increment if we didn't remove
    }
  }

  // Second pass: create new dummy bonds for complementary pairs within cutoff
  
  // Get property arrays
  int *hyb_status = atom->ivector[hyb_status_index];
  int *size = atom->ivector[size_index];
  
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    
    if (!(mask[i] & groupbit)) continue;
    
    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;
      
      jtype = type[j];
      // Atoms must be opposite types (one type 1, one type 2)
      if (!((itype == iatomtype && jtype == jatomtype) || 
            (itype == jatomtype && jtype == iatomtype))) continue;
      
      if (i == j) continue;

      // Check distance with minimum image convention
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      if (domain->xperiodic) {
        if (delx > domain->xprd_half) delx -= domain->xprd;
        else if (delx < -domain->xprd_half) delx += domain->xprd;
      }
      if (domain->yperiodic) {
        if (dely > domain->yprd_half) dely -= domain->yprd;
        else if (dely < -domain->yprd_half) dely += domain->yprd;
      }
      if (domain->zperiodic) {
        if (delz > domain->zprd_half) delz -= domain->zprd;
        else if (delz < -domain->zprd_half) delz += domain->zprd;
      }
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq > cutoffsq) continue;

      // Check if complementary
      jtag = tag[j];
      if (!is_complementary(itag, jtag)) continue;

      // Check if dummy bond already exists
      if (dummy_bond_exists(i, j)) continue;
      
      // Check capacity constraint
      int min_size = (size[i] < size[j]) ? size[i] : size[j];
      if (hyb_status[i] + min_size > 2) continue;
      if (hyb_status[j] + min_size > 2) continue;

      // Only create bond on lower-tagged atom (with newton_bond on)
      if (itag > jtag) continue;

      if (num_bond[i] >= atom->bond_per_atom)
        error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/pre");

      // Create the dummy bond
      bond_type[i][num_bond[i]] = dummy_btype;
      bond_atom[i][num_bond[i]] = jtag;
      num_bond[i]++;
      createcount++;
    }
  }

  // Accumulate counts across processors
  int createcountall, removecountall;
  MPI_Allreduce(&createcount, &createcountall, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&removecount, &removecountall, 1, MPI_INT, MPI_SUM, world);
  
  createcounttotal += createcountall;
  removecounttotal += removecountall;
  createcount = createcountall;
  removecount = removecountall;

  // If any bonds were created or removed, rebuild special lists and trigger reneighboring
  if (createcount > 0 || removecount > 0) {
    next_reneighbor = update->ntimestep;
    Special special(lmp);
    special.build();
  }
}

double FixDnafoldBondPre::compute_vector(int n)
{
  if (n == 0) return (double) createcount;
  if (n == 1) return (double) removecount;
  if (n == 2) return (double) createcounttotal;
  return (double) removecounttotal;
}

double FixDnafoldBondPre::memory_usage()
{
  // Estimate set memory (rough approximation)
  double bytes = complementarity_set.size() * (2 * sizeof(tagint) + 32);
  return bytes;
}

