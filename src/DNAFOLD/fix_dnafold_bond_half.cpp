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

#include "fix_dnafold_bond_half.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "special.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixDnafoldBondHalf::FixDnafoldBondHalf(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), bond_requests(nullptr)
{
  if (narg != 6) error->all(FLERR,"Illegal fix dnafold/bond/half command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/bond/half nevery");

  double cutoff = utils::numeric(FLERR,arg[4],false,lmp);
  if (cutoff <= 0.0) error->all(FLERR,"Illegal fix dnafold/bond/half cutoff");
  cutoff_sq = cutoff * cutoff;

  bond_type = utils::inumeric(FLERR,arg[5],false,lmp);
  if (bond_type <= 0) error->all(FLERR,"Illegal fix dnafold/bond/half bond_type");

  type1 = 1;
  type2 = 2;

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  
  vector_flag = 1;
  size_vector = 4;  // create, break, total create, total break
  global_freq = 1;
  extvector = 0;

  createcount = 0;
  breakcount = 0;
  createcounttotal = 0;
  breakcounttotal = 0;

  nmax = 0;
  maxrequest = 0;
  num_requests = 0;

  comm_forward = 0;
  comm_reverse = 3;  // tag1, tag2, bond_type
}

/* ---------------------------------------------------------------------- */

FixDnafoldBondHalf::~FixDnafoldBondHalf()
{
  memory->destroy(bond_requests);
}

/* ---------------------------------------------------------------------- */

int FixDnafoldBondHalf::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHalf::init()
{
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/bond/half with non-molecular system");
  
  if (force->bond == nullptr)
    error->all(FLERR,"Must define bond_style for fix dnafold/bond/half");

  if (bond_type > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix dnafold/bond/half");

  if (force->newton_bond == 0)
    error->all(FLERR,"Fix dnafold/bond/half requires newton bond on");

  // Find the i_size property
  int flag_size, cols_size;
  size_index = atom->find_custom("size", flag_size, cols_size);
  if (size_index < 0)
    error->all(FLERR,"Could not find i_size property for fix dnafold/bond/half");
  if (flag_size != 0)
    error->all(FLERR,"Property i_size must be an integer property");
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHalf::setup(int /* vflag */)
{
  post_integrate();
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHalf::post_integrate()
{
  if (update->ntimestep % nevery) return;

  // Acquire updated ghost atom positions
  comm->forward_comm();

  int nlocal = atom->nlocal;
  
  createcount = 0;
  breakcount = 0;
  num_requests = 0;

  // First: break stretched bonds
  break_stretched_bonds();

  // Second: create new bonds between same-type atoms
  create_same_type_bonds();

  // Communicate bond creation requests to ghost home processors
  if (num_requests > 0) {
    comm->reverse_comm(this);
  }

  // Accumulate counts across processors
  int createcountall, breakcountall;
  MPI_Allreduce(&createcount, &createcountall, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&breakcount, &breakcountall, 1, MPI_INT, MPI_SUM, world);
  
  createcounttotal += createcountall;
  breakcounttotal += breakcountall;
  createcount = createcountall;
  breakcount = breakcountall;

  // If any bonds were created or removed, rebuild special lists and trigger reneighboring
  if (createcount > 0 || breakcount > 0) {
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

    if (me == 0) {
      if (screen_save)
        fprintf(screen_save,"Fix dnafold/bond/half: created %d, broke %d bonds at step " BIGINT_FORMAT "\n",
                createcount, breakcount, update->ntimestep);
      if (logfile_save)
        fprintf(logfile_save,"Fix dnafold/bond/half: created %d, broke %d bonds at step " BIGINT_FORMAT "\n",
                createcount, breakcount, update->ntimestep);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHalf::create_same_type_bonds()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type_arr = atom->bond_type;
  int *size = atom->ivector[size_index];
  double **x = atom->x;

  // Loop over local atoms as central atoms
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    
    int itype = type[i];
    if (itype != type1 && itype != type2) continue;

    // Central atom must have size = 2
    if (size[i] != 2) continue;

    // Collect all opposite-type bonded neighbors with size = 1
    std::vector<int> opposite_neighbors;
    
    for (int n = 0; n < nspecial[i][0]; n++) {
      tagint ntag = special[i][n];
      int nloc = atom->map(ntag);
      
      if (nloc < 0) continue;  // Not on this processor
      if (!(mask[nloc] & groupbit)) continue;
      
      int ntype = type[nloc];
      
      // Check if opposite type
      if ((itype == type1 && ntype == type2) || (itype == type2 && ntype == type1)) {
        // Check if neighbor has size = 1
        if (size[nloc] == 1) {
          opposite_neighbors.push_back(nloc);
        }
      }
    }
    
    // Check all pairs of opposite-type neighbors (both size 1)
    for (size_t a = 0; a < opposite_neighbors.size(); a++) {
      for (size_t b = a+1; b < opposite_neighbors.size(); b++) {
        int j = opposite_neighbors[a];
        int k = opposite_neighbors[b];
        
        tagint jtag = tag[j];
        tagint ktag = tag[k];
        
        // j and k are both bonded to i, are the same type (opposite of i), and both size 1
        // Check if they're already bonded to each other
        if (atoms_bonded(j, k)) continue;
        
        // Check distance between j and k (with minimum image)
        double delx = x[j][0] - x[k][0];
        double dely = x[j][1] - x[k][1];
        double delz = x[j][2] - x[k][2];
        
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
        
        double rsq = delx*delx + dely*dely + delz*delz;
        if (rsq > cutoff_sq) continue;
        
        // Determine which is lower tagged
        tagint lower_tag = (jtag < ktag) ? jtag : ktag;
        tagint higher_tag = (jtag < ktag) ? ktag : jtag;
        int lower_local = (jtag < ktag) ? j : k;
        
        // Create bond on lower-tagged atom
        if (lower_local < nlocal) {
          // Lower-tagged atom is local - create directly
          if (num_bond[lower_local] >= atom->bond_per_atom)
            error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/half");
          
          bond_type_arr[lower_local][num_bond[lower_local]] = bond_type;
          bond_atom[lower_local][num_bond[lower_local]] = higher_tag;
          num_bond[lower_local]++;
          createcount++;
        } else {
          // Lower-tagged atom is ghost - queue for reverse comm
          if (num_requests >= maxrequest) {
            maxrequest += 100;
            memory->grow(bond_requests, maxrequest, 3, "fix_dnafold_bond_half:bond_requests");
          }
          bond_requests[num_requests][0] = lower_tag;
          bond_requests[num_requests][1] = higher_tag;
          bond_requests[num_requests][2] = bond_type;
          num_requests++;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHalf::break_stretched_bonds()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *num_bond = atom->num_bond;
  int **bond_type_arr = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *type = atom->type;
  int *size = atom->ivector[size_index];
  
  for (int i = 0; i < nlocal; i++) {
    int k = 0;
    while (k < num_bond[i]) {
      if (bond_type_arr[i][k] != bond_type) {
        k++;
        continue;
      }
      
      // Atom i must be a half-bead (size == 1)
      if (size[i] != 1) {
        k++;
        continue;
      }
      
      // Find the partner atom
      tagint j_tag = bond_atom[i][k];
      int j = atom->map(j_tag);
      
      if (j < 0) {
        // Partner atom not found - remove bond
        num_bond[i]--;
        bond_atom[i][k] = bond_atom[i][num_bond[i]];
        bond_type_arr[i][k] = bond_type_arr[i][num_bond[i]];
        breakcount++;
        continue;  // Don't increment k since we shifted
      }
      
      // Partner must also be a half-bead (size == 1) and same type as i
      if (size[j] != 1 || type[j] != type[i]) {
        k++;
        continue;
      }
      
      // Calculate distance with minimum image convention
      double delx = x[i][0] - x[j][0];
      double dely = x[i][1] - x[j][1];
      double delz = x[i][2] - x[j][2];
      
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
      
      double rsq = delx*delx + dely*dely + delz*delz;
      
      if (rsq > cutoff_sq) {
        // Break this bond - remove from atom i
        num_bond[i]--;
        bond_atom[i][k] = bond_atom[i][num_bond[i]];
        bond_type_arr[i][k] = bond_type_arr[i][num_bond[i]];
        breakcount++;
        // Don't increment k since we moved the last bond into this position
      } else {
        k++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixDnafoldBondHalf::atoms_bonded(int i, int j)
{
  tagint *tag = atom->tag;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;

  tagint jtag = tag[j];

  // Check if j is in i's 1-2 neighbor list
  for (int k = 0; k < nspecial[i][0]; k++) {
    if (special[i][k] == jtag) return 1;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixDnafoldBondHalf::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  
  for (int i = 0; i < num_requests; i++) {
    buf[m++] = ubuf(bond_requests[i][0]).d;
    buf[m++] = ubuf(bond_requests[i][1]).d;
    buf[m++] = ubuf(bond_requests[i][2]).d;
  }
  
  return m;
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHalf::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  tagint *tag = atom->tag;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type_arr = atom->bond_type;
  int nlocal = atom->nlocal;
  
  // Receive bond creation requests
  int num_recv = n / 3;
  
  for (int i = 0; i < num_recv; i++) {
    tagint tag1 = (tagint) ubuf(buf[m++]).i;
    tagint tag2 = (tagint) ubuf(buf[m++]).i;
    int btype = (int) ubuf(buf[m++]).i;
    
    // Find local atom with tag1 (should be local since we sent it here)
    int iloc = atom->map(tag1);
    
    if (iloc < 0 || iloc >= nlocal) continue;  // Safety check
    
    // Check if bond already exists
    int already_exists = 0;
    for (int k = 0; k < num_bond[iloc]; k++) {
      if (bond_atom[iloc][k] == tag2 && bond_type_arr[iloc][k] == btype) {
        already_exists = 1;
        break;
      }
    }
    
    if (already_exists) continue;
    
    // Create the bond
    if (num_bond[iloc] >= atom->bond_per_atom)
      error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/half");
    
    bond_type_arr[iloc][num_bond[iloc]] = btype;
    bond_atom[iloc][num_bond[iloc]] = tag2;
    num_bond[iloc]++;
    createcount++;
  }
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondHalf::compute_vector(int n)
{
  if (n == 0) return (double) createcount;
  if (n == 1) return (double) breakcount;
  if (n == 2) return (double) createcounttotal;
  return (double) breakcounttotal;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondHalf::memory_usage()
{
  double bytes = (double)maxrequest * 3 * sizeof(tagint);
  return bytes;
}

