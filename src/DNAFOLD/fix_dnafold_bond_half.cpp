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

/* ----------------------------------------------------------------------
   DNAFOLD package: Coarse-grained DNA origami folding simulation

   fix dnafold/bond/half creates and breaks "half-bonds" between
   same-type half-beads (size=1) that share a common central whole-bead
   (size=2). These bonds connect half-beads on opposite strands that
   are each bonded to the same central bead, helping maintain the
   geometry of partially hybridized regions.
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
  // syntax: fix ID group dnafold/bond/half nevery cutoff bond_type
  if (narg != 6) error->all(FLERR,"Illegal fix dnafold/bond/half command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // parse nevery - how often to check for bond creation/breaking
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/bond/half nevery");

  // parse cutoff distance for half-bond creation and breaking
  double cutoff = utils::numeric(FLERR,arg[4],false,lmp);
  if (cutoff <= 0.0) error->all(FLERR,"Illegal fix dnafold/bond/half cutoff");
  cutoff_sq = cutoff * cutoff;

  // parse the bond type to use for half-bonds
  half_bond_type = utils::inumeric(FLERR,arg[5],false,lmp);
  if (half_bond_type <= 0) error->all(FLERR,"Illegal fix dnafold/bond/half bond_type");

  // half-bonds form between same-type atoms (both type 1 or both type 2)
  // but we search from central atoms that are type 1 or type 2
  iatomtype = 1;
  jatomtype = 2;

  // set up fix flags for reneighboring and output vector
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  vector_flag = 1;
  size_vector = 4;  // create, break, total create, total break
  global_freq = 1;
  extvector = 0;

  // initialize counters
  create_count = 0;
  break_count = 0;
  create_count_total = 0;
  break_count_total = 0;

  // initialize communication arrays
  nmax = 0;
  max_requests = 0;
  num_requests = 0;

  // reverse comm sends 3 values per request: tag1, tag2, bond_type
  comm_forward = 0;
  comm_reverse = 3;
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
  // validate system is molecular
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/bond/half with non-molecular system");

  // validate bond style is defined
  if (force->bond == nullptr)
    error->all(FLERR,"Must define bond_style for fix dnafold/bond/half");

  // validate half bond type is within range
  if (half_bond_type > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix dnafold/bond/half");

  // require newton bond on for proper bond storage
  if (force->newton_bond == 0)
    error->all(FLERR,"Fix dnafold/bond/half requires newton bond on");

  // find the size custom property (1 for half-beads, 2 for whole beads)
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

/* ----------------------------------------------------------------------
   Main function called every nevery timesteps.
   Two phases: (1) break stretched bonds, (2) create new bonds.
------------------------------------------------------------------------- */

void FixDnafoldBondHalf::post_integrate()
{
  if (update->ntimestep % nevery) return;

  // acquire updated ghost atom positions
  comm->forward_comm();

  int nlocal = atom->nlocal;

  // reset per-step counters
  create_count = 0;
  break_count = 0;
  num_requests = 0;

  // phase 1: break half-bonds that have stretched beyond cutoff
  break_stretched_bonds();

  // phase 2: create new half-bonds between eligible same-type neighbors
  create_same_type_bonds();

  // send bond creation requests to ghost atoms' home processors
  if (num_requests > 0) {
    comm->reverse_comm(this);
  }

  // accumulate counts across all MPI processors
  int create_count_all, break_count_all;
  MPI_Allreduce(&create_count, &create_count_all, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&break_count, &break_count_all, 1, MPI_INT, MPI_SUM, world);

  create_count_total += create_count_all;
  break_count_total += break_count_all;
  create_count = create_count_all;
  break_count = break_count_all;

  // if any bonds changed, rebuild special neighbor lists and trigger reneighboring
  if (create_count > 0 || break_count > 0) {
    // suppress verbose output from Special::build()
    FILE *screen_save = screen;
    FILE *logfile_save = logfile;
    screen = nullptr;
    logfile = nullptr;

    Special special(lmp);
    special.build();
    comm->borders();

    // restore output streams
    screen = screen_save;
    logfile = logfile_save;

    next_reneighbor = update->ntimestep;
  }
}

/* ----------------------------------------------------------------------
   Create half-bonds between same-type half-beads that share a common
   central whole-bead. We loop over central atoms (size=2) and look for
   pairs of opposite-type neighbors (relative to center) with size=1.
------------------------------------------------------------------------- */

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
  int **bond_type = atom->bond_type;
  int *size = atom->ivector[size_index];
  double **x = atom->x;

  // loop over local atoms as potential central atoms
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    int itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    // central atom must be a whole bead (size = 2)
    if (size[i] != 2) continue;

    // collect all bonded neighbors that are:
    // - opposite type from central atom
    // - half-beads (size = 1)
    std::vector<int> opposite_neighbors;

    for (int n = 0; n < nspecial[i][0]; n++) {
      tagint ntag = special[i][n];
      int nloc = atom->map(ntag);

      if (nloc < 0) continue;  // neighbor not on this processor
      if (!(mask[nloc] & groupbit)) continue;

      int ntype = type[nloc];

      // check if neighbor is opposite type from central atom
      if ((itype == iatomtype && ntype == jatomtype) || (itype == jatomtype && ntype == iatomtype)) {
        // check if neighbor is a half-bead
        if (size[nloc] == 1) {
          opposite_neighbors.push_back(nloc);
        }
      }
    }

    // check all pairs of opposite-type neighbors for potential half-bond creation
    // these pairs have the same type as each other (both opposite from center)
    for (size_t a = 0; a < opposite_neighbors.size(); a++) {
      for (size_t b = a+1; b < opposite_neighbors.size(); b++) {
        int j = opposite_neighbors[a];
        int k = opposite_neighbors[b];

        tagint jtag = tag[j];
        tagint ktag = tag[k];

        // skip if j and k are already bonded to each other
        if (are_atoms_bonded(j, k)) continue;

        // calculate distance between j and k with minimum image convention
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

        // skip if beyond cutoff distance
        if (rsq > cutoff_sq) continue;

        // determine which atom has the lower tag for consistent bond storage
        tagint lower_tag = (jtag < ktag) ? jtag : ktag;
        tagint higher_tag = (jtag < ktag) ? ktag : jtag;
        int lower_local = (jtag < ktag) ? j : k;

        // create bond on lower-tagged atom (newton bond convention)
        if (lower_local < nlocal) {
          // lower-tagged atom is local - create bond directly
          if (num_bond[lower_local] >= atom->bond_per_atom)
            error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/half");

          bond_type[lower_local][num_bond[lower_local]] = half_bond_type;
          bond_atom[lower_local][num_bond[lower_local]] = higher_tag;
          num_bond[lower_local]++;
          create_count++;
        } else {
          // lower-tagged atom is a ghost - queue request for reverse communication
          if (num_requests >= max_requests) {
            max_requests += 100;
            memory->grow(bond_requests, max_requests, 3, "fix_dnafold_bond_half:bond_requests");
          }
          bond_requests[num_requests][0] = lower_tag;
          bond_requests[num_requests][1] = higher_tag;
          bond_requests[num_requests][2] = half_bond_type;
          num_requests++;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Break half-bonds that have stretched beyond the cutoff distance.
   Only processes bonds of type half_bond_type between half-beads.
------------------------------------------------------------------------- */

void FixDnafoldBondHalf::break_stretched_bonds()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *type = atom->type;
  int *size = atom->ivector[size_index];

  // loop over local atoms
  for (int i = 0; i < nlocal; i++) {
    int k = 0;

    // loop through bonds using while loop since we may remove bonds
    while (k < num_bond[i]) {
      // skip if not a half-bond
      if (bond_type[i][k] != half_bond_type) {
        k++;
        continue;
      }

      // atom i must be a half-bead (size == 1) for this to be a valid half-bond
      if (size[i] != 1) {
        k++;
        continue;
      }

      // find the partner atom
      tagint j_tag = bond_atom[i][k];
      int j = atom->map(j_tag);

      // partner not found - remove the bond
      if (j < 0) {
        num_bond[i]--;
        bond_atom[i][k] = bond_atom[i][num_bond[i]];
        bond_type[i][k] = bond_type[i][num_bond[i]];
        break_count++;
        continue;  // don't increment k since we shifted
      }

      // partner must also be a half-bead (size == 1) and same type as i
      if (size[j] != 1 || type[j] != type[i]) {
        k++;
        continue;
      }

      // calculate distance with minimum image convention
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

      // break bond if stretched beyond cutoff
      if (rsq > cutoff_sq) {
        num_bond[i]--;
        bond_atom[i][k] = bond_atom[i][num_bond[i]];
        bond_type[i][k] = bond_type[i][num_bond[i]];
        break_count++;
        // don't increment k since we moved the last bond into this position
      } else {
        k++;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Check if two atoms are bonded using the special neighbor list.
   Returns 1 if bonded (j is in i's 1-2 neighbor list), 0 otherwise.
------------------------------------------------------------------------- */

int FixDnafoldBondHalf::are_atoms_bonded(int i, int j)
{
  tagint jtag = atom->tag[j];
  tagint *slist = atom->special[i];
  int n1 = atom->nspecial[i][0];

  // check if j is in i's 1-2 (directly bonded) neighbor list
  for (int k = 0; k < n1; k++) {
    if (slist[k] == jtag) return 1;
  }

  return 0;
}

/* ----------------------------------------------------------------------
   Pack bond creation requests for reverse communication.
   Sends requests to home processors of ghost atoms.
------------------------------------------------------------------------- */

int FixDnafoldBondHalf::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;

  // pack all queued bond requests into buffer
  for (int i = 0; i < num_requests; i++) {
    buf[m++] = ubuf(bond_requests[i][0]).d;  // tag1 (lower)
    buf[m++] = ubuf(bond_requests[i][1]).d;  // tag2 (higher)
    buf[m++] = ubuf(bond_requests[i][2]).d;  // bond type
  }

  return m;
}

/* ----------------------------------------------------------------------
   Unpack bond creation requests from reverse communication.
   Creates bonds on local atoms that were requested by other processors.
------------------------------------------------------------------------- */

void FixDnafoldBondHalf::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  tagint *tag = atom->tag;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int nlocal = atom->nlocal;

  // number of bond requests received
  int num_recv = n / 3;

  // process each received bond creation request
  for (int i = 0; i < num_recv; i++) {
    tagint tag1 = (tagint) ubuf(buf[m++]).i;
    tagint tag2 = (tagint) ubuf(buf[m++]).i;
    int btype = (int) ubuf(buf[m++]).i;

    // find local atom with tag1 (should be local since request was sent here)
    int iloc = atom->map(tag1);

    // safety check: must be a local atom
    if (iloc < 0 || iloc >= nlocal) continue;

    // check if this bond already exists (avoid duplicates)
    int already_exists = 0;
    for (int k = 0; k < num_bond[iloc]; k++) {
      if (bond_atom[iloc][k] == tag2 && bond_type[iloc][k] == btype) {
        already_exists = 1;
        break;
      }
    }

    if (already_exists) continue;

    // create the bond
    if (num_bond[iloc] >= atom->bond_per_atom)
      error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/half");

    bond_type[iloc][num_bond[iloc]] = btype;
    bond_atom[iloc][num_bond[iloc]] = tag2;
    num_bond[iloc]++;
    create_count++;
  }
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondHalf::compute_vector(int n)
{
  // output vector: [0]=created, [1]=broken, [2]=total_created, [3]=total_broken
  if (n == 0) return (double) create_count;
  if (n == 1) return (double) break_count;
  if (n == 2) return (double) create_count_total;
  return (double) break_count_total;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondHalf::memory_usage()
{
  double bytes = (double)max_requests * 3 * sizeof(tagint);
  return bytes;
}
