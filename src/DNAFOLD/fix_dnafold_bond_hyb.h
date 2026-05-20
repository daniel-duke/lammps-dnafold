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
FixStyle(dnafold/bond/hyb,FixDnafoldBondHyb);
// clang-format on
#else

#ifndef LMP_FIX_DNAFOLD_BOND_HYB_H
#define LMP_FIX_DNAFOLD_BOND_HYB_H

#include "fix.h"

#include <unordered_map>
#include <vector>

namespace LAMMPS_NS {

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
  // === MPI info ===
  int me, nprocs;

  // === Atom type info ===
  int iatomtype, jatomtype;

  // === Bond parameters ===
  int dummy_bond_type;
  double cutoff_sq;
  char *complementarity_file;

  // === Property indices ===
  int hyb_status_5p_index;
  int hyb_status_3p_index;

  // === Temperature data ===
  std::vector<double> temperatures;
  int num_temperatures;
  char *tvar;
  int tvar_index;

  // === Complementarity data ===
  struct PairData {
    int half_i;
    int half_j;
    std::vector<double> energies;
  };
  struct PairHash {
    std::size_t operator()(const std::pair<tagint, tagint> &p) const {
      return std::hash<tagint>()(p.first) ^ (std::hash<tagint>()(p.second) << 1);
    }
  };
  std::unordered_map<std::pair<tagint, tagint>, PairData, PairHash> complementarity_map;
  std::vector<std::pair<double, int>> energy_levels;

  // === Counters ===
  int create_count, downgrade_count;
  int update_count;
  bigint create_count_total, downgrade_count_total;

  // === Communication arrays ===
  int nmax;
  tagint *partner;
  tagint *final_partner;
  int *partner_bond_type;
  double *dist_sq;
  double *partner_energy;
  int commflag;

  // === Per-timestep shared state ===
  struct HybStatusChange {
    tagint tag;
    int delta_5p;
    int delta_3p;
  };
  std::vector<HybStatusChange> hyb_status_changes;

  // === Helper functions ===
  void downgrade_bonds();
  void upgrade_bonds();
  void broadcast_hyb_status_changes();
  void read_complementarity_file();
  double get_interpolated_energy(tagint, tagint);
  int get_bond_type(tagint, tagint);
std::pair<int,int> get_halves(tagint, tagint);
  void remove_dummy_bond(int, int);
  bool is_hyb_bond_type(int btype);
};

}    // namespace LAMMPS_NS

#endif
#endif
