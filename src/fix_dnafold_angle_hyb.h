/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(dnafold/angle/hyb,FixDnafoldAngleHyb);
// clang-format on
#else

#ifndef LMP_FIX_DNAFOLD_ANGLE_HYB_H
#define LMP_FIX_DNAFOLD_ANGLE_HYB_H

#include "fix.h"

namespace LAMMPS_NS {

static constexpr double EPSILON = 1.0e-10;  // Tolerance for floating point comparisons

class FixDnafoldAngleHyb : public Fix {
 public:
  FixDnafoldAngleHyb(class LAMMPS *, int, char **);
  ~FixDnafoldAngleHyb() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_integrate() override;
  double compute_vector(int) override;
  double memory_usage() override;

 private:
  int me, nprocs;
  int hyb_status_index;          // index for d_hyb_status in atom->dvector
  int is_crossover_index;        // index for d_is_crossover in atom->dvector
  int size_index;                // index for d_size in atom->dvector
  int createcount;               // angles created this timestep
  bigint createcounttotal;       // cumulative angles created

  void find_and_create_angles(); // search for and create angles
  int angle_exists(int, int, int);  // check if angle already exists
  int atoms_bonded(int, int);    // check if two atoms are bonded
};

}    // namespace LAMMPS_NS

#endif
#endif


