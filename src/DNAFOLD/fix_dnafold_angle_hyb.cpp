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

   fix dnafold/angle/hyb creates angles between fully hybridized atoms.
   Angles are created when a central atom and two of its bonded neighbors
   are all fully hybridized (hyb_status == size). The angle type depends
   on whether the central atom is at a crossover junction (is_crossover).
   Angles are stored on the central atom.
------------------------------------------------------------------------- */

#include "fix_dnafold_angle_hyb.h"

#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "special.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixDnafoldAngleHyb::FixDnafoldAngleHyb(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  // syntax: fix ID group dnafold/angle/hyb nevery max_angle_deviation
  if (narg != 5) error->all(FLERR,"Illegal fix dnafold/angle/hyb command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // parse nevery - how often to check for angle creation
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/angle/hyb command");

  // parse max_angle_deviation - maximum deviation from equilibrium angle in degrees
  max_angle_deviation = utils::numeric(FLERR,arg[4],false,lmp);
  if (max_angle_deviation <= 0.0 || max_angle_deviation > 90.0)
    error->all(FLERR,"Illegal max_angle_deviation for fix dnafold/angle/hyb (must be 0 < max_angle_deviation <= 90)");

  // set up fix flags for reneighboring and output vector
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  // initialize counters
  create_count = 0;
  create_count_total = 0;
}

/* ---------------------------------------------------------------------- */

FixDnafoldAngleHyb::~FixDnafoldAngleHyb()
{
}

/* ---------------------------------------------------------------------- */

int FixDnafoldAngleHyb::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDnafoldAngleHyb::init()
{
  // find the hyb_status custom property (tracks hybridization slots used: 0-2)
  int flag_hyb, cols_hyb;
  hyb_status_index = atom->find_custom("hyb_status", flag_hyb, cols_hyb);
  if (hyb_status_index < 0)
    error->all(FLERR,"Could not find i_hyb_status property for fix dnafold/angle/hyb");
  if (flag_hyb != 0)
    error->all(FLERR,"Property i_hyb_status must be an integer property");

  // find the is_crossover custom property (1 if at crossover junction, 0 otherwise)
  int flag_cross, cols_cross;
  is_crossover_index = atom->find_custom("is_crossover", flag_cross, cols_cross);
  if (is_crossover_index < 0)
    error->all(FLERR,"Could not find i_is_crossover property for fix dnafold/angle/hyb");
  if (flag_cross != 0)
    error->all(FLERR,"Property i_is_crossover must be an integer property");

  // find the size custom property (1 for half-beads, 2 for whole beads)
  int flag_size, cols_size;
  size_index = atom->find_custom("size", flag_size, cols_size);
  if (size_index < 0)
    error->all(FLERR,"Could not find i_size property for fix dnafold/angle/hyb");
  if (flag_size != 0)
    error->all(FLERR,"Property i_size must be an integer property");

  // validate system is molecular
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/angle/hyb with non-molecular system");

  // validate angle style is defined
  if (force->angle == nullptr)
    error->all(FLERR,"Cannot use fix dnafold/angle/hyb without angle_style defined");
}

/* ---------------------------------------------------------------------- */

void FixDnafoldAngleHyb::setup(int /* vflag */)
{
  post_integrate();
}

/* ----------------------------------------------------------------------
   Main function called every nevery timesteps.
   Finds fully hybridized atom triplets and creates angles.
------------------------------------------------------------------------- */

void FixDnafoldAngleHyb::post_integrate()
{
  if (update->ntimestep % nevery) return;

  // find and create angles for fully hybridized triplets
  find_and_create_angles();

  // accumulate counts across all MPI processors
  int create_count_all;
  MPI_Allreduce(&create_count,&create_count_all,1,MPI_INT,MPI_SUM,world);
  create_count_total += create_count_all;
  create_count = create_count_all;

  // if any angles were created, rebuild special neighbor lists and trigger reneighboring
  // (angles affect 1-3 special interactions and pairwise exclusions)
  if (create_count > 0) {
    // suppress verbose output from Special::build()
    FILE *screen_save = screen;
    FILE *logfile_save = logfile;
    screen = nullptr;
    logfile = nullptr;

    Special special(lmp);
    special.build();

    // restore output streams
    screen = screen_save;
    logfile = logfile_save;

    // trigger neighbor list rebuild to update pairwise exclusions
    next_reneighbor = update->ntimestep;
  }
}

/* ----------------------------------------------------------------------
   Find fully hybridized atom triplets and create angles.
   An angle j-i-k is created when:
   - Center atom i is fully hybridized (hyb_status[i] == size[i])
   - Both end atoms j and k are bonded to i and fully hybridized
   - Atoms j and k are not directly bonded to each other
   - The angle doesn't already exist
------------------------------------------------------------------------- */

void FixDnafoldAngleHyb::find_and_create_angles()
{
  int i,j,k;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  int *tag = atom->tag;
  int **angle_type = atom->angle_type;
  int **angle_atom1 = atom->angle_atom1;
  int **angle_atom2 = atom->angle_atom2;
  int **angle_atom3 = atom->angle_atom3;
  int *num_angle = atom->num_angle;

  // get custom property arrays
  int *hyb_status = atom->ivector[hyb_status_index];
  int *is_crossover = atom->ivector[is_crossover_index];
  int *size = atom->ivector[size_index];

  // get special neighbor lists (1-2 = directly bonded atoms)
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;

  // reset per-step counter
  create_count = 0;

  // loop over local atoms as potential center atoms
  for (i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;

    // only create angles for fully hybridized atoms
    // fully hybridized means hyb_status equals the atom's size
    if (hyb_status[i] != size[i]) continue;

    tagint itag = tag[i];

    // angle type depends on whether center atom is at a crossover junction
    // type 1 = normal angle, type 2 = crossover angle
    int atype = 1 + is_crossover[i];

    // validate angle type is within range
    if (atype <= 0 || atype > atom->nangletypes)
      error->one(FLERR,"Fix dnafold/angle/hyb: Invalid angle type calculated from is_crossover");

    // collect bonded neighbors that are also fully hybridized
    std::vector<int> valid_neighbors;

    for (int n = 0; n < nspecial[i][0]; n++) {
      tagint ntag = special[i][n];
      int nloc = atom->map(ntag);

      if (nloc < 0) continue;  // neighbor not on this processor
      if (!(mask[nloc] & groupbit)) continue;

      // neighbor must also be fully hybridized
      if (hyb_status[nloc] != size[nloc]) continue;

      valid_neighbors.push_back(nloc);
    }

    // need at least 2 valid neighbors to form an angle
    if (valid_neighbors.size() < 2) continue;

    // check all pairs of valid neighbors for angle creation
    for (size_t j_idx = 0; j_idx < valid_neighbors.size(); j_idx++) {
      int jlocal = valid_neighbors[j_idx];
      tagint jtag = tag[jlocal];

      for (size_t k_idx = j_idx+1; k_idx < valid_neighbors.size(); k_idx++) {
        int klocal = valid_neighbors[k_idx];
        tagint ktag = tag[klocal];

        // skip if same atom (shouldn't happen, but safety check)
        if (jtag == ktag) continue;

        // skip if j and k are directly bonded (would form degenerate angle)
        if (are_atoms_bonded(jlocal, klocal)) continue;

        // skip if angle already exists
        if (has_angle(jlocal, i, klocal)) continue;

        // skip if current angle deviates too much from equilibrium
        // equilibrium: 180° for type 1 (normal), 90° for type 2 (crossover)
        double current_angle = compute_angle(jlocal, i, klocal);
        double equilibrium_angle = (atype == 1) ? 180.0 : 90.0;
        double deviation = fabs(current_angle - equilibrium_angle);
        if (deviation > max_angle_deviation) continue;

        // create angle j-i-k on center atom i
        if (num_angle[i] >= atom->angle_per_atom)
          error->one(FLERR,"Fix dnafold/angle/hyb: Too many angles per atom");

        angle_type[i][num_angle[i]] = atype;
        angle_atom1[i][num_angle[i]] = jtag;
        angle_atom2[i][num_angle[i]] = itag;
        angle_atom3[i][num_angle[i]] = ktag;
        num_angle[i]++;

        create_count++;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Check if two atoms are directly bonded using the special neighbor list.
   Returns 1 if bonded (j is in i's 1-2 neighbor list), 0 otherwise.
------------------------------------------------------------------------- */

int FixDnafoldAngleHyb::are_atoms_bonded(int i, int j)
{
  tagint jtag = atom->tag[j];
  tagint *slist = atom->special[i];
  int n1 = atom->nspecial[i][0];

  // check if j is in i's 1-2 (directly bonded) neighbor list
  // this is globally consistent and handles newton bond storage
  for (int k = 0; k < n1; k++) {
    if (slist[k] == jtag) return 1;
  }

  return 0;
}

/* ----------------------------------------------------------------------
   Check if angle i-j-k already exists on center atom j.
   Angles are stored on the center atom, so we only need to check j's list.
------------------------------------------------------------------------- */

int FixDnafoldAngleHyb::has_angle(int i, int j, int k)
{
  int *tag = atom->tag;
  int **angle_atom1 = atom->angle_atom1;
  int **angle_atom3 = atom->angle_atom3;
  int *num_angle = atom->num_angle;

  tagint itag = tag[i];
  tagint ktag = tag[k];

  // check if angle i-j-k already exists on center atom j
  // angles are symmetric: i-j-k is the same as k-j-i
  for (int m = 0; m < num_angle[j]; m++) {
    if ((angle_atom1[j][m] == itag && angle_atom3[j][m] == ktag) ||
        (angle_atom1[j][m] == ktag && angle_atom3[j][m] == itag)) {
      return 1;
    }
  }

  return 0;
}

/* ----------------------------------------------------------------------
   Compute the angle j-i-k in degrees (i is the center atom).
   Uses minimum image convention for periodic boundaries.
------------------------------------------------------------------------- */

double FixDnafoldAngleHyb::compute_angle(int j, int i, int k)
{
  double **x = atom->x;

  // vectors from center atom i to end atoms j and k
  double delx1 = x[j][0] - x[i][0];
  double dely1 = x[j][1] - x[i][1];
  double delz1 = x[j][2] - x[i][2];

  double delx2 = x[k][0] - x[i][0];
  double dely2 = x[k][1] - x[i][1];
  double delz2 = x[k][2] - x[i][2];

  // apply minimum image convention for vector 1
  if (domain->xperiodic) {
    if (delx1 > domain->xprd_half) delx1 -= domain->xprd;
    else if (delx1 < -domain->xprd_half) delx1 += domain->xprd;
  }
  if (domain->yperiodic) {
    if (dely1 > domain->yprd_half) dely1 -= domain->yprd;
    else if (dely1 < -domain->yprd_half) dely1 += domain->yprd;
  }
  if (domain->zperiodic) {
    if (delz1 > domain->zprd_half) delz1 -= domain->zprd;
    else if (delz1 < -domain->zprd_half) delz1 += domain->zprd;
  }

  // apply minimum image convention for vector 2
  if (domain->xperiodic) {
    if (delx2 > domain->xprd_half) delx2 -= domain->xprd;
    else if (delx2 < -domain->xprd_half) delx2 += domain->xprd;
  }
  if (domain->yperiodic) {
    if (dely2 > domain->yprd_half) dely2 -= domain->yprd;
    else if (dely2 < -domain->yprd_half) dely2 += domain->yprd;
  }
  if (domain->zperiodic) {
    if (delz2 > domain->zprd_half) delz2 -= domain->zprd;
    else if (delz2 < -domain->zprd_half) delz2 += domain->zprd;
  }

  // compute magnitudes
  double r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);
  double r2 = sqrt(delx2*delx2 + dely2*dely2 + delz2*delz2);

  if (r1 < 1e-10 || r2 < 1e-10) return 0.0;  // degenerate case

  // compute cos(angle) via dot product
  double cos_angle = (delx1*delx2 + dely1*dely2 + delz1*delz2) / (r1 * r2);

  // clamp to [-1, 1] for numerical safety
  if (cos_angle > 1.0) cos_angle = 1.0;
  if (cos_angle < -1.0) cos_angle = -1.0;

  // return angle in degrees
  return acos(cos_angle) * 180.0 / M_PI;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldAngleHyb::compute_vector(int n)
{
  // output vector: [0]=created, [1]=total_created
  if (n == 0) return (double) create_count;
  return (double) create_count_total;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldAngleHyb::memory_usage()
{
  return 0.0;
}
