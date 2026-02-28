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
FixStyle(dnafold/bond/half,FixDnafoldBondHalf);
// clang-format on
#else

#ifndef LMP_FIX_DNAFOLD_BOND_HALF_H
#define LMP_FIX_DNAFOLD_BOND_HALF_H

#include "fix.h"

namespace LAMMPS_NS {

class FixDnafoldBondHalf : public Fix {
 public:
  FixDnafoldBondHalf(class LAMMPS *, int, char **);
  ~FixDnafoldBondHalf() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_integrate() override;
  double compute_vector(int) override;
  double memory_usage() override;

 private:
  // === MPI info ===
  int me, nprocs;

  // === Atom type info ===
  int iatomtype, jatomtype;

  // === Bond parameters ===
  int half_bond_type;
  double cutoff_sq;

  // === Property indices ===
  int size_index;

  // === Counters ===
  // Output vector: [0]=created, [1]=broken, [2]=total_created, [3]=total_broken
  int create_count, break_count;
  bigint create_count_total, break_count_total;

  // === Communication arrays ===
  int nmax;
  int max_requests;
  int num_requests;
  tagint **bond_requests;

  // === Helper functions ===
  void create_same_type_bonds();
  void break_stretched_bonds();
  bool has_bond(int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
