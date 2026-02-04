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
FixStyle(dnafold/bond/pre,FixDnafoldBondPre);
// clang-format on
#else

#ifndef LMP_FIX_DNAFOLD_BOND_PRE_H
#define LMP_FIX_DNAFOLD_BOND_PRE_H

#include "fix.h"
#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

// Hash function for pair of tagints
struct PairHashPre {
  std::size_t operator()(const std::pair<tagint, tagint> &p) const {
    return std::hash<tagint>()(p.first) ^ (std::hash<tagint>()(p.second) << 1);
  }
};

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
  int me, nprocs;
  int iatomtype, jatomtype;      // atom types (1 and 2)
  int dummy_btype;               // dummy bond type
  double cutoffsq;               // distance cutoff squared
  char *complementarity_file;    // file containing complementarity pairs
  int hyb_status_index;          // index for i_hyb_status in atom->ivector
  int size_index;                // index for i_size in atom->ivector
  int createcount;               // bonds created this timestep
  int removecount;               // bonds removed this timestep
  bigint createcounttotal;       // cumulative bonds created
  bigint removecounttotal;       // cumulative bonds removed

  class NeighList *list;         // neighbor list

  // Temperature-dependent complementarity data
  std::vector<double> temperatures;     // list of temperatures from file
  int num_temperatures;                  // number of temperature points
  double min_energy_threshold;           // minimum energy for bond formation (from TYPES)

  // Map of complementary pairs: (tag1, tag2) -> vector of energies at each temperature
  std::unordered_map<std::pair<tagint, tagint>, std::vector<double>, PairHashPre> complementarity_map;

  // Temperature variable access
  char *tvar;                            // name of temperature variable (without v_ prefix)
  int tvar_index;                        // index of temperature variable

  void read_complementarity_file();      // read complementarity pairs from file
  double get_interpolated_energy(tagint, tagint); // get energy at current temperature
  bool is_complementary(tagint, tagint); // check if pair is complementary at current T
  bool dummy_bond_exists(int, int);      // check if dummy bond exists between two atoms
  bool any_bond_exists(int, int);        // check if any bond exists between two atoms
};

}    // namespace LAMMPS_NS

#endif
#endif

