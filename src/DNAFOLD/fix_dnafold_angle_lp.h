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
   DNAFOLD package: Coarse-grained DNA origami folding simulation
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(dnafold/angle/lp,FixDnafoldAngleLp);
// clang-format on
#else

#ifndef LMP_FIX_DNAFOLD_ANGLE_LP_H
#define LMP_FIX_DNAFOLD_ANGLE_LP_H

#include "fix.h"
#include <vector>

namespace LAMMPS_NS {

class FixDnafoldAngleLp : public Fix {
 public:
  FixDnafoldAngleLp(class LAMMPS *, int, char **);
  ~FixDnafoldAngleLp() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_integrate() override;
  double memory_usage() override;

 private:
  // === MPI info ===
  int me, nprocs;

  // === Input parameters ===
  double r12;                      // characteristic length for conversion
  char *lp_file;                   // persistence length file path
  char *tvar;                      // temperature variable name (without v_)
  int tvar_index;                  // temperature variable index

  // === Persistence length data (from file) ===
  std::vector<double> temperatures;
  std::vector<double> persistence_lengths;
  int num_data_points;

  // === Angle parameter access ===
  double *k_angle;                 // pointer to angle K array (from extract)

  // === Constants ===
  // Boltzmann constant in nano units (pN·nm/K)
  static constexpr double BOLTZMANN = 0.01380649;

  // === Helper functions ===
  void read_lp_file();
  double get_persistence_length(double T);
  void update_angle_k();
};

}    // namespace LAMMPS_NS

#endif
#endif
