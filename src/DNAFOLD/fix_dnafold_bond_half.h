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
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 private:
  int me, nprocs;
  int type1, type2;              // atom types (1 and 2)
  int bond_type;                 // type of bond to create/break
  double cutoff_sq;              // squared cutoff distance for breaking bonds
  int size_index;                // index for i_size in atom->ivector
  int createcount;               // bonds created this timestep
  int breakcount;                // bonds broken this timestep
  bigint createcounttotal;       // cumulative bonds created
  bigint breakcounttotal;        // cumulative bonds broken

  // Arrays for reverse communication of bond requests
  int nmax;                      // size of per-atom arrays
  int maxrequest;                // size of bond_requests array
  int num_requests;              // number of bond requests to send
  tagint **bond_requests;        // array of [tag1, tag2, bond_type] requests

  void create_same_type_bonds(); // find and create bonds between same-type atoms
  void break_stretched_bonds();  // break bonds that exceed cutoff
  int atoms_bonded(int, int);    // check if two atoms are bonded
};

}    // namespace LAMMPS_NS

#endif
#endif

