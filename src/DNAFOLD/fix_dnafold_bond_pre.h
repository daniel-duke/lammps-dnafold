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
FixStyle(dnafold/bond/pre,FixDnafoldBondPre);
// clang-format on
#else

#ifndef LMP_FIX_DNAFOLD_BOND_PRE_H
#define LMP_FIX_DNAFOLD_BOND_PRE_H

#include "fix.h"

#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

class FixDnafoldBondPre : public Fix {
 public:
  FixDnafoldBondPre(class LAMMPS *, int, char **);
  ~FixDnafoldBondPre() override;

  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
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
  int dummy_bond_type;
  double cutoff_sq;
  char *complementarity_file;

  // === Property indices ===
  int hyb_status_index;
  int size_index;

  // === Temperature data ===
  std::vector<double> temperatures;
  int num_temperatures;
  double min_energy_threshold;
  char *tvar;
  int tvar_index;

  // === Complementarity data ===
  // Hash function for pair of tagints
  struct PairHash {
    std::size_t operator()(const std::pair<tagint, tagint> &p) const {
      return std::hash<tagint>()(p.first) ^ (std::hash<tagint>()(p.second) << 1);
    }
  };
  // Map: (tag1, tag2) -> vector of energies at each temperature
  // tag1 < tag2 always for consistent lookup
  std::unordered_map<std::pair<tagint, tagint>, std::vector<double>, PairHash> complementarity_map;

  // === Counters ===
  // Output vector: [0]=created, [1]=removed, [2]=total_created, [3]=total_removed
  int create_count, remove_count;
  bigint create_count_total, remove_count_total;

  // === Neighbor list ===
  class NeighList *list;

  // === Helper functions ===
  void read_complementarity_file();
  double get_interpolated_energy(tagint, tagint);
  bool is_complementary(tagint, tagint);
  bool has_bond(int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
