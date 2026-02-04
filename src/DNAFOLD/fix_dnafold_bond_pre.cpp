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
#include "special.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "variable.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double BIG = 1.0e20;

FixDnafoldBondPre::FixDnafoldBondPre(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), complementarity_file(nullptr), tvar(nullptr)
{
  if (narg != 8) error->all(FLERR,"Illegal fix dnafold/bond/pre command");

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

  // Parse temperature variable (must start with v_)
  if (strncmp(arg[7], "v_", 2) != 0)
    error->all(FLERR,"Temperature variable for fix dnafold/bond/pre must start with v_");
  tvar = utils::strdup(arg[7] + 2);  // skip "v_" prefix
  tvar_index = -1;  // will be set in init()

  num_temperatures = 0;
  min_energy_threshold = 0.0;

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
  delete[] tvar;
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

  // Find and validate temperature variable
  tvar_index = input->variable->find(tvar);
  if (tvar_index < 0)
    error->all(FLERR,"Variable {} for fix dnafold/bond/pre does not exist", tvar);
  if (!input->variable->equalstyle(tvar_index))
    error->all(FLERR,"Variable {} for fix dnafold/bond/pre must be equal-style", tvar);

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
  // Temporary storage for TYPES section to find minimum threshold
  std::vector<double> type_energies;

  // Only rank 0 reads the file
  if (me == 0) {
    std::ifstream file(complementarity_file);
    if (!file.is_open())
      error->one(FLERR,fmt::format("Cannot open complementarity file '{}'", complementarity_file));

    std::string line;
    int line_num = 0;
    enum Section { NONE, TEMPERATURES, TYPES, PAIRS };
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
      if (trimmed == "TEMPERATURES") {
        current_section = TEMPERATURES;
        continue;
      } else if (trimmed == "TYPES") {
        current_section = TYPES;
        continue;
      } else if (trimmed == "PAIRS") {
        current_section = PAIRS;
        if (temperatures.empty()) {
          error->one(FLERR,fmt::format("TEMPERATURES section must appear before PAIRS in '{}'",
                                       complementarity_file));
        }
        continue;
      }

      // Parse TEMPERATURES section
      if (current_section == TEMPERATURES) {
        std::istringstream iss(line);
        double temp;
        while (iss >> temp) {
          temperatures.push_back(temp);
        }
        if (temperatures.empty()) {
          error->one(FLERR,fmt::format("Invalid TEMPERATURES format in '{}' at line {}",
                                       complementarity_file, line_num));
        }
        num_temperatures = temperatures.size();
        continue;
      }

      // Parse TYPES section to find minimum energy threshold
      if (current_section == TYPES) {
        std::istringstream iss(line);
        int btype;
        double energy;

        if (!(iss >> btype >> energy)) {
          error->one(FLERR,fmt::format("Invalid TYPES format in '{}' at line {}",
                                       complementarity_file, line_num));
        }
        type_energies.push_back(energy);
        continue;
      }

      // Parse PAIRS section
      if (current_section == PAIRS) {
        std::istringstream iss(line);
        tagint tag1, tag2;

        if (!(iss >> tag1 >> tag2)) {
          error->one(FLERR,fmt::format("Invalid PAIRS format in '{}' at line {}",
                                       complementarity_file, line_num));
        }

        // Read energy values for each temperature
        std::vector<double> energies;
        double energy;
        while (iss >> energy) {
          energies.push_back(energy);
        }

        if ((int)energies.size() != num_temperatures) {
          error->one(FLERR,fmt::format("PAIRS line {} in '{}' has {} energies, expected {}",
                                       line_num, complementarity_file,
                                       energies.size(), num_temperatures));
        }

        // Ensure tag1 < tag2 for consistent lookup
        if (tag1 > tag2) std::swap(tag1, tag2);

        // Store in map
        complementarity_map[std::make_pair(tag1, tag2)] = energies;

      } else if (current_section == NONE) {
        error->one(FLERR,fmt::format("Line {} in '{}' appears before any section header",
                                     line_num, complementarity_file));
      }
    }

    file.close();

    if (temperatures.empty()) {
      error->one(FLERR,fmt::format("No TEMPERATURES section found in '{}'", complementarity_file));
    }

    if (type_energies.empty()) {
      error->one(FLERR,fmt::format("No TYPES section found in '{}'", complementarity_file));
    }

    // Find minimum energy threshold (lowest energy in TYPES = weakest bond threshold)
    min_energy_threshold = *std::min_element(type_energies.begin(), type_energies.end());

    if (complementarity_map.empty()) {
      error->warning(FLERR,"Complementarity file '{}' contains no valid pairs", complementarity_file);
    }
  }

  // Broadcast num_temperatures and temperatures
  MPI_Bcast(&num_temperatures, 1, MPI_INT, 0, world);

  if (num_temperatures > 0) {
    if (me != 0) temperatures.resize(num_temperatures);
    MPI_Bcast(temperatures.data(), num_temperatures, MPI_DOUBLE, 0, world);
  }

  // Broadcast min_energy_threshold
  MPI_Bcast(&min_energy_threshold, 1, MPI_DOUBLE, 0, world);

  // Broadcast map size
  int map_size = complementarity_map.size();
  MPI_Bcast(&map_size, 1, MPI_INT, 0, world);

  // Broadcast map contents
  if (map_size > 0) {
    std::vector<tagint> tags1(map_size);
    std::vector<tagint> tags2(map_size);
    std::vector<double> all_energies(map_size * num_temperatures);

    if (me == 0) {
      int idx = 0;
      for (const auto &entry : complementarity_map) {
        tags1[idx] = entry.first.first;
        tags2[idx] = entry.first.second;
        for (int t = 0; t < num_temperatures; t++) {
          all_energies[idx * num_temperatures + t] = entry.second[t];
        }
        idx++;
      }
    }

    MPI_Bcast(tags1.data(), map_size, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(tags2.data(), map_size, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(all_energies.data(), map_size * num_temperatures, MPI_DOUBLE, 0, world);

    if (me != 0) {
      for (int i = 0; i < map_size; i++) {
        std::vector<double> energies(num_temperatures);
        for (int t = 0; t < num_temperatures; t++) {
          energies[t] = all_energies[i * num_temperatures + t];
        }
        complementarity_map[std::make_pair(tags1[i], tags2[i])] = energies;
      }
    }
  }

  if (me == 0) {
    if (screen)
      fprintf(screen,"Fix dnafold/bond/pre: Read %d temperatures and %d complementarity pairs from '%s'\n",
              num_temperatures, map_size, complementarity_file);
    if (logfile)
      fprintf(logfile,"Fix dnafold/bond/pre: Read %d temperatures and %d complementarity pairs from '%s'\n",
              num_temperatures, map_size, complementarity_file);
  }
}

double FixDnafoldBondPre::get_interpolated_energy(tagint tag_i, tagint tag_j)
{
  // Ensure tag1 < tag2 for consistent lookup
  tagint tag1 = (tag_i < tag_j) ? tag_i : tag_j;
  tagint tag2 = (tag_i < tag_j) ? tag_j : tag_i;

  auto it = complementarity_map.find(std::make_pair(tag1, tag2));
  if (it == complementarity_map.end()) return -BIG;  // Not complementary

  const std::vector<double>& energies = it->second;

  // Get current temperature from variable
  double T = input->variable->compute_equal(tvar_index);

  // Check bounds - error if outside range
  if (T < temperatures.front() || T > temperatures.back()) {
    error->all(FLERR, fmt::format(
      "Fix dnafold/bond/pre: Temperature {} outside complementarity file range [{}, {}]",
      T, temperatures.front(), temperatures.back()));
  }

  // Find bracketing temperatures and interpolate
  for (int i = 0; i < num_temperatures - 1; i++) {
    if (T >= temperatures[i] && T <= temperatures[i+1]) {
      double t0 = temperatures[i];
      double t1 = temperatures[i+1];
      double e0 = energies[i];
      double e1 = energies[i+1];
      double frac = (T - t0) / (t1 - t0);
      return e0 + frac * (e1 - e0);
    }
  }

  return energies.back();  // Should not reach here, but return last value
}

bool FixDnafoldBondPre::is_complementary(tagint tag_i, tagint tag_j)
{
  double energy = get_interpolated_energy(tag_i, tag_j);
  // Pair is complementary if energy >= minimum threshold
  return energy >= min_energy_threshold;
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

bool FixDnafoldBondPre::any_bond_exists(int i, int j)
{
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  tagint jtag = atom->tag[j];

  // Check if i has any bond to j (regardless of bond type)
  for (int k = 0; k < num_bond[i]; k++) {
    if (bond_atom[i][k] == jtag) return true;
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

  // First pass: remove dummy bonds that are too far apart or whose energy dropped below threshold
  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    itag = tag[i];

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
      
      // Check that atoms are opposite types
      jtype = type[j];
      if (!((itype == iatomtype && jtype == jatomtype) || 
            (itype == jatomtype && jtype == iatomtype))) {
        k++;
        continue;
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

      // Check if should remove: too far OR energy below threshold at current T
      bool should_remove = false;
      if (rsq > cutoffsq) {
        should_remove = true;
      } else {
        // Check if energy dropped below threshold
        if (!is_complementary(itag, jtag)) {
          should_remove = true;
        }
      }

      if (should_remove) {
        // Remove bond - shift remaining bonds down
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

      // Check if any bond already exists (dummy, hyb, or other)
      if (any_bond_exists(i, j)) continue;
      
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
    // Suppress output from Special::build()
    FILE *screen_save = screen;
    FILE *logfile_save = logfile;
    screen = nullptr;
    logfile = nullptr;

    Special special(lmp);
    special.build();

    // Restore output
    screen = screen_save;
    logfile = logfile_save;

    next_reneighbor = update->ntimestep;
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
  // Estimate map memory (rough approximation)
  // Each entry: 2 tagints for key + vector of num_temperatures doubles + overhead
  double bytes = complementarity_map.size() * (2 * sizeof(tagint) + num_temperatures * sizeof(double) + 32);
  // Add temperatures vector
  bytes += temperatures.size() * sizeof(double);
  return bytes;
}

