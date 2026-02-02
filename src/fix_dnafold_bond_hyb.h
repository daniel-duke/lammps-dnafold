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
FixStyle(dnafold/bond/hyb,FixDnafoldBondHyb);
// clang-format on
#else

#ifndef LMP_FIX_DNAFOLD_BOND_HYB_H
#define LMP_FIX_DNAFOLD_BOND_HYB_H

#include "fix.h"
#include <unordered_map>

namespace LAMMPS_NS {

// Hash function for pair of tagints
struct PairHash {
  std::size_t operator()(const std::pair<tagint, tagint> &p) const {
    return std::hash<tagint>()(p.first) ^ (std::hash<tagint>()(p.second) << 1);
  }
};

class FixDnafoldBondHyb : public Fix {
 public:
  FixDnafoldBondHyb(class LAMMPS *, int, char **);
  ~FixDnafoldBondHyb() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_integrate() override;
  double compute_vector(int) override;
  double memory_usage() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 private:
  int me, nprocs;
  int iatomtype, jatomtype;      // atom types (1 and 2)
  int dummy_btype;               // dummy bond type to ignore/remove
  double cutoffsq;               // distance cutoff squared
  char *complementarity_file;    // file containing complementarity pairs and bond types
  int property_flag_index;       // index for i_hyb_status in atom->ivector
  int size_index;                // index for i_size in atom->ivector
  int createcount;               // bonds created this timestep
  int downgradecount;            // bonds downgraded this timestep
  bigint createcounttotal;       // cumulative bonds created
  bigint downgradecounttotal;    // cumulative bonds downgraded

  // Complementarity map: (tag1, tag2) -> energy_depth
  // tag1 < tag2 always
  std::unordered_map<std::pair<tagint, tagint>, double, PairHash> complementarity_map;
  
  // Energy levels: sorted pairs of (energy_depth, bond_type)
  // Sorted in descending order by energy_depth
  std::vector<std::pair<double, int>> energy_levels;

  // Arrays for partner selection across processors
  int nmax;                      // size of per-atom arrays
  tagint *partner;               // tag of bond partner for each atom
  tagint *finalpartner;          // final partner after communication
  int *partnerbtype;             // bond type for the partner
  double *distsq;                // distance squared with partner
  int commflag;                  // flag for communication mode

  void read_complementarity_file();  // read complementarity pairs from file
  int get_bond_type(tagint, tagint); // get bond type for a pair (returns 0 if not complementary)
  int bond_exists(int, int);         // check if bond exists between two atoms
  void remove_dummy_bond(int, int);  // remove dummy bond between two atoms
};

}    // namespace LAMMPS_NS

#endif
#endif

