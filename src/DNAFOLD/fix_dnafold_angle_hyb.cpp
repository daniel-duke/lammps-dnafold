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
   Contributing author: Custom DNA folding simulation fix
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
  if (narg != 4) error->all(FLERR,"Illegal fix dnafold/angle/hyb command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/angle/hyb command");

  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  createcount = 0;
  createcounttotal = 0;
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
  // Find the i_hyb_status property (now an integer)
  int flag_hyb, cols_hyb;
  hyb_status_index = atom->find_custom("hyb_status", flag_hyb, cols_hyb);
  if (hyb_status_index < 0)
    error->all(FLERR,"Could not find i_hyb_status property for fix dnafold/angle/hyb");
  if (flag_hyb != 0)
    error->all(FLERR,"Property i_hyb_status must be an integer property");

  // Find the i_is_crossover property (now an integer)
  int flag_cross, cols_cross;
  is_crossover_index = atom->find_custom("is_crossover", flag_cross, cols_cross);
  if (is_crossover_index < 0)
    error->all(FLERR,"Could not find i_is_crossover property for fix dnafold/angle/hyb");
  if (flag_cross != 0)
    error->all(FLERR,"Property i_is_crossover must be an integer property");

  // Find the i_size property (now an integer)
  int flag_size, cols_size;
  size_index = atom->find_custom("size", flag_size, cols_size);
  if (size_index < 0)
    error->all(FLERR,"Could not find i_size property for fix dnafold/angle/hyb");
  if (flag_size != 0)
    error->all(FLERR,"Property i_size must be an integer property");

  // Verify system supports angles
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/angle/hyb with non-molecular system");
  
  if (force->angle == nullptr)
    error->all(FLERR,"Cannot use fix dnafold/angle/hyb without angle_style defined");
}

/* ---------------------------------------------------------------------- */

void FixDnafoldAngleHyb::setup(int /* vflag */)
{
  post_integrate();
}

/* ---------------------------------------------------------------------- */

void FixDnafoldAngleHyb::post_integrate()
{
  if (update->ntimestep % nevery) return;

  find_and_create_angles();

  // Sum angle creation across processors
  int createcountall;
  MPI_Allreduce(&createcount,&createcountall,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcountall;
  createcount = createcountall;

  // If any angles were created, rebuild special lists
  // (angles affect 1-3 special interactions)
  if (createcount > 0) {
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
  }
}

/* ---------------------------------------------------------------------- */

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
  
  // Get property arrays as integers
  int *hyb_status = atom->ivector[hyb_status_index];
  int *is_crossover = atom->ivector[is_crossover_index];
  int *size = atom->ivector[size_index];
  
  // Get special bond arrays
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;

  createcount = 0;

  // Loop over potential center atoms
  for (i = 0; i < nlocal; i++) {
    
    if (!(mask[i] & groupbit)) continue;
    
    // Only create angles if hyb_status equals size (fully hybridized)
    if (hyb_status[i] != size[i]) continue;

    tagint itag = tag[i];
    
    // Calculate angle type for this central atom: atype = 1 + is_crossover[i]
    int atype = 1 + is_crossover[i];
    
    // Verify angle type is valid
    if (atype <= 0 || atype > atom->nangletypes)
      error->one(FLERR,"Fix dnafold/angle/hyb: Invalid angle type calculated from is_crossover");
    
    // Get bonded neighbors from special bond list
    // special[i][0 ... nspecial[i][0]-1] contains directly bonded atoms
    std::vector<int> valid_neighbors;
    
    for (int n = 0; n < nspecial[i][0]; n++) {
      tagint ntag = special[i][n];
      int nloc = atom->map(ntag);
      
      if (nloc < 0) continue;
      if (!(mask[nloc] & groupbit)) continue;
      
      // Neighbor must also be fully hybridized
      if (hyb_status[nloc] != size[nloc]) continue;
      
      valid_neighbors.push_back(nloc);
    }
    
    if (valid_neighbors.size() < 2) continue;

    // Check all pairs of neighbors
    for (size_t j_idx = 0; j_idx < valid_neighbors.size(); j_idx++) {
      int jlocal = valid_neighbors[j_idx];
      tagint jtag = tag[jlocal];
      
      for (size_t k_idx = j_idx+1; k_idx < valid_neighbors.size(); k_idx++) {
        int klocal = valid_neighbors[k_idx];
        tagint ktag = tag[klocal];
        
        if (jtag == ktag) continue;
        
        // Check that j and k are not bonded to each other
        if (atoms_bonded(jlocal, klocal)) continue;

        // Check if angle already exists
        if (angle_exists(jlocal, i, klocal)) continue;

        // Create angle j-i-k with angle type based on central atom's is_crossover
        if (num_angle[i] >= atom->angle_per_atom)
          error->one(FLERR,"Fix dnafold/angle/hyb: Too many angles per atom");

        angle_type[i][num_angle[i]] = atype;
        angle_atom1[i][num_angle[i]] = jtag;
        angle_atom2[i][num_angle[i]] = itag;
        angle_atom3[i][num_angle[i]] = ktag;
        num_angle[i]++;

        createcount++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixDnafoldAngleHyb::atoms_bonded(int i, int j)
{
  int *tag = atom->tag;
  tagint **special = atom->special;
  int **nspecial = atom->nspecial;

  tagint jtag = tag[j];

  // Check special bond list (1-2 neighbors = directly bonded atoms)
  // This is globally consistent and handles newton bond storage
  for (int k = 0; k < nspecial[i][0]; k++) {
    if (special[i][k] == jtag) return 1;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixDnafoldAngleHyb::angle_exists(int i, int j, int k)
{
  int *tag = atom->tag;
  int **angle_atom1 = atom->angle_atom1;
  int **angle_atom3 = atom->angle_atom3;
  int *num_angle = atom->num_angle;

  tagint itag = tag[i];
  tagint ktag = tag[k];

  // Check if angle i-j-k already exists on center atom j
  // Angles are stored on the center atom, so we only check atom j's angle list
  for (int m = 0; m < num_angle[j]; m++) {
    if ((angle_atom1[j][m] == itag && angle_atom3[j][m] == ktag) ||
        (angle_atom1[j][m] == ktag && angle_atom3[j][m] == itag)) {
      return 1;
    }
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldAngleHyb::compute_vector(int n)
{
  if (n == 0) return (double) createcount;
  return (double) createcounttotal;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldAngleHyb::memory_usage()
{
  return 0.0;
}


