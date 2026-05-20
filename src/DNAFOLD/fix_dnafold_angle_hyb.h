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

/* ----------------------------------------------------------------------
   DNAFOLD package: Mesoscopic DNA origami folding simulation
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
  // === MPI info ===
  int me, nprocs;

  // === Property indices ===
  int hyb_status_5p_index;
  int hyb_status_3p_index;
  int is_crossover_index;
  int size_index;

  // === Angle constraint ===
  double max_angle_deviation;

  // === Counters ===
  int create_count;
  bigint create_count_total;

  // === Helper functions ===
  void create_angles();
  bool has_angle(int, int, int);
  bool has_bond(int, int);
  double compute_angle(int, int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
