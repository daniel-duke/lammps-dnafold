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
   DNAFOLD package: Coarse-grained DNA origami folding simulation

   fix dnafold/bond/hyb manages the hybridization state of DNA bonds.
   It upgrades dummy bonds (precursor bonds from fix dnafold/bond/pre)
   to hybridization bonds when atoms are within cutoff, and downgrades
   hybridization bonds back to dummy bonds when they stretch too far or
   when the temperature-dependent binding energy drops below threshold.
   Bond types are selected based on interpolated binding energies from
   a complementarity file.
------------------------------------------------------------------------- */

#include "fix_dnafold_bond_hyb.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "special.h"
#include "update.h"
#include "variable.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double BIG = 1.0e20;
static constexpr int MAX_HYB_CAPACITY = 2;

/* ---------------------------------------------------------------------- */

FixDnafoldBondHyb::FixDnafoldBondHyb(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), complementarity_file(nullptr), tvar(nullptr),
  partner(nullptr), final_partner(nullptr), partner_bond_type(nullptr),
  dist_sq(nullptr), partner_energy(nullptr)
{
  // syntax: fix ID group dnafold/bond/hyb nevery cutoff dummy_btype compfile tempvar
  if (narg != 8) error->all(FLERR,"Illegal fix dnafold/bond/hyb command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // parse nevery - how often to check for bond upgrades/downgrades
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/bond/hyb command");

  // set up fix flags for reneighboring and output vector
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  vector_flag = 1;
  size_vector = 4;
  global_freq = 1;
  extvector = 0;

  // forward/reverse comm sends 4 values: partner tag, bond type, distance squared, energy depth
  comm_forward = 4;
  comm_reverse = 4;

  // hybridization bonds form between type 1 (scaffold) and type 2 (staple) atoms
  iatomtype = 1;
  jatomtype = 2;

  // parse cutoff distance for hybridization
  cutoff_sq = utils::numeric(FLERR,arg[4],false,lmp);
  if (cutoff_sq <= 0.0) error->all(FLERR,"Illegal cutoff");
  cutoff_sq = cutoff_sq * cutoff_sq;

  // parse dummy bond type (these get upgraded to hyb bonds)
  dummy_bond_type = utils::inumeric(FLERR,arg[5],false,lmp);
  if (dummy_bond_type <= 0) error->all(FLERR,"Illegal dummy bond type");

  // parse complementarity file path
  complementarity_file = utils::strdup(arg[6]);

  // parse temperature variable name (must start with v_)
  if (strncmp(arg[7], "v_", 2) != 0)
    error->all(FLERR,"Temperature variable for fix dnafold/bond/hyb must start with v_");
  tvar = utils::strdup(arg[7] + 2);
  tvar_index = -1;

  // initialize counters
  create_count = 0;
  downgrade_count = 0;
  create_count_total = 0;
  downgrade_count_total = 0;

  // initialize data structures
  nmax = 0;
  num_temperatures = 0;

  // read complementarity data from file
  read_complementarity_file();
}

/* ---------------------------------------------------------------------- */

FixDnafoldBondHyb::~FixDnafoldBondHyb()
{
  delete[] complementarity_file;
  delete[] tvar;
  memory->destroy(partner);
  memory->destroy(final_partner);
  memory->destroy(partner_bond_type);
  memory->destroy(dist_sq);
  memory->destroy(partner_energy);
}

/* ---------------------------------------------------------------------- */

int FixDnafoldBondHyb::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHyb::init()
{
  // find the hyb_status custom property (tracks hybridization slots used: 0-2)
  int flag_hyb, cols_hyb;
  hyb_status_index = atom->find_custom("hyb_status", flag_hyb, cols_hyb);
  if (hyb_status_index < 0)
    error->all(FLERR,fmt::format("Could not find property 'i_hyb_status'"));
  if (flag_hyb != 0)
    error->all(FLERR,fmt::format("Property 'i_hyb_status' must be integer"));

  // find the size custom property (1 for half-beads, 2 for whole beads)
  int flag_size, cols_size;
  size_index = atom->find_custom("size", flag_size, cols_size);
  if (size_index < 0)
    error->all(FLERR,fmt::format("Could not find property 'i_size'"));
  if (flag_size != 0)
    error->all(FLERR,fmt::format("Property 'i_size' must be integer"));

  // find and validate the temperature variable
  tvar_index = input->variable->find(tvar);
  if (tvar_index < 0)
    error->all(FLERR,"Variable {} for fix dnafold/bond/hyb does not exist", tvar);
  if (!input->variable->equalstyle(tvar_index))
    error->all(FLERR,"Variable {} for fix dnafold/bond/hyb must be equal-style", tvar);

  // validate system is molecular
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/bond/hyb with non-molecular system");

  // validate bond style is defined
  if (force->bond == nullptr)
    error->all(FLERR,"Must define bond_style");

  // require newton bond on for proper bond storage
  if (force->newton_bond == 0)
    error->all(FLERR,"Fix dnafold/bond/hyb requires newton bond on");

  // validate dummy bond type is within range
  if (dummy_bond_type > atom->nbondtypes)
    error->all(FLERR,"Invalid dummy bond type in fix dnafold/bond/hyb");
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondHyb::setup(int /* vflag */)
{
  post_integrate();
}

/* ----------------------------------------------------------------------
   Read complementarity file containing temperature-dependent binding energies.
   File format has three sections:
   - TYPES: bond type to energy depth mapping (determines which bond type to use)
   - TEMPS: list of temperature values for interpolation
   - PAIRS: atom tag pairs with binding energies at each temperature
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::read_complementarity_file()
{
  // rank 0 reads the file, then broadcasts data to all processors
  if (me == 0) {
    std::ifstream file(complementarity_file);
    if (!file.is_open())
      error->one(FLERR,fmt::format("Cannot open complementarity file '{}'", complementarity_file));

    std::string line;
    int line_num = 0;
    enum Section { NONE, TYPES, TEMPS, PAIRS };
    Section current_section = NONE;

    // parse file line by line, handling each section differently
    while (std::getline(file, line)) {
      line_num++;

      // skip empty lines and comment lines starting with #
      if (line.empty()) continue;
      if (line[0] == '#') continue;

      // trim leading whitespace for section header detection
      size_t start = line.find_first_not_of(" \t");
      if (start == std::string::npos) continue;
      std::string trimmed = line.substr(start);

      // check for section headers and switch parsing mode
      if (trimmed == "TYPES") {
        current_section = TYPES;
        continue;
      } else if (trimmed == "TEMPS") {
        current_section = TEMPS;
        continue;
      } else if (trimmed == "PAIRS") {
        current_section = PAIRS;
        if (temperatures.empty()) {
          error->one(FLERR,fmt::format("TEMPS section must appear before PAIRS in '{}'",
                                       complementarity_file));
        }
        continue;
      }

      // parse TYPES section: read bond type and energy depth pairs
      // energy_levels maps energy depths to bond types for bond type selection
      if (current_section == TYPES) {
        std::istringstream iss(line);
        int btype;
        double energy;

        if (!(iss >> btype >> energy)) {
          error->one(FLERR,fmt::format("Invalid TYPES format in '{}' at line {}",
                                       complementarity_file, line_num));
        }

        if (btype <= 0) {
          error->one(FLERR,fmt::format("Invalid bond type {} in '{}' at line {}",
                                       btype, complementarity_file, line_num));
        }

        // store energy level and bond type pair
        energy_levels.push_back(std::make_pair(energy, btype));
        continue;
      }

      // parse TEMPS section: read list of temperature values (in Celsius)
      if (current_section == TEMPS) {
        std::istringstream iss(line);
        double temp;
        while (iss >> temp) {
          temperatures.push_back(temp + 273.15);  // convert Celsius to Kelvin
        }
        if (temperatures.empty()) {
          error->one(FLERR,fmt::format("Invalid TEMPS format in '{}' at line {}",
                                       complementarity_file, line_num));
        }
        num_temperatures = temperatures.size();
        continue;
      }

      // parse PAIRS section: read complementary atom pairs with per-temperature energies
      // format: tag1 tag2 <ignored_column> energy1 energy2 ...
      if (current_section == PAIRS) {
        std::istringstream iss(line);
        tagint tag1, tag2;
        std::string ignored_column;

        if (!(iss >> tag1 >> tag2 >> ignored_column)) {
          error->one(FLERR,fmt::format("Invalid PAIRS format in '{}' at line {}",
                                       complementarity_file, line_num));
        }

        // read energy values for each temperature point
        std::vector<double> energies;
        double energy;
        while (iss >> energy) {
          energies.push_back(energy);
        }

        // verify we have the right number of energy values
        if ((int)energies.size() != num_temperatures) {
          error->one(FLERR,fmt::format("PAIRS line {} in '{}' has {} energies, expected {}",
                                       line_num, complementarity_file,
                                       energies.size(), num_temperatures));
        }

        // store with tag1 < tag2 for consistent lookup
        if (tag1 > tag2) std::swap(tag1, tag2);
        complementarity_map[std::make_pair(tag1, tag2)] = energies;

      } else if (current_section == NONE) {
        error->one(FLERR,fmt::format("Line {} in '{}' appears before any section header",
                                     line_num, complementarity_file));
      }
    }

    file.close();

    // validate required sections were found
    if (temperatures.empty()) {
      error->one(FLERR,fmt::format("No TEMPS section found in '{}'", complementarity_file));
    }

    if (energy_levels.empty()) {
      error->one(FLERR,fmt::format("No bond types defined in TYPES section of '{}'",
                                   complementarity_file));
    }

    if (complementarity_map.empty()) {
      error->warning(FLERR,"Complementarity file '{}' contains no valid pairs",
                     complementarity_file);
    }

    // sort energy levels by energy (descending) for efficient bond type lookup
    // higher energy = stronger bond
    std::sort(energy_levels.begin(), energy_levels.end(),
              [](const std::pair<double,int> &a, const std::pair<double,int> &b) {
                return a.first > b.first;  // descending order
              });

    // sort temperatures (ascending) and reorder pair energies to match
    // this allows temperatures to be listed in any order in the file
    if (num_temperatures > 1) {
      // create index array and sort by temperature
      std::vector<size_t> sort_indices(num_temperatures);
      for (int i = 0; i < num_temperatures; i++) sort_indices[i] = i;

      std::sort(sort_indices.begin(), sort_indices.end(),
                [this](size_t a, size_t b) { return temperatures[a] < temperatures[b]; });

      // reorder temperatures
      std::vector<double> sorted_temps(num_temperatures);
      for (int i = 0; i < num_temperatures; i++) {
        sorted_temps[i] = temperatures[sort_indices[i]];
      }
      temperatures = std::move(sorted_temps);

      // reorder each pair's energy vector using the same permutation
      for (auto &entry : complementarity_map) {
        std::vector<double> sorted_energies(num_temperatures);
        for (int i = 0; i < num_temperatures; i++) {
          sorted_energies[i] = entry.second[sort_indices[i]];
        }
        entry.second = std::move(sorted_energies);
      }
    }
  }

  // ========== Broadcast data to all processors ==========

  // broadcast temperature array
  MPI_Bcast(&num_temperatures, 1, MPI_INT, 0, world);

  if (num_temperatures > 0) {
    if (me != 0) temperatures.resize(num_temperatures);
    MPI_Bcast(temperatures.data(), num_temperatures, MPI_DOUBLE, 0, world);
  }

  // broadcast energy levels (bond type to energy mapping)
  int num_levels = energy_levels.size();
  MPI_Bcast(&num_levels, 1, MPI_INT, 0, world);

  if (num_levels > 0) {
    if (me != 0) energy_levels.resize(num_levels);

    // pack into flat arrays for MPI broadcast
    std::vector<double> level_energies(num_levels);
    std::vector<int> btypes(num_levels);

    if (me == 0) {
      for (int i = 0; i < num_levels; i++) {
        level_energies[i] = energy_levels[i].first;
        btypes[i] = energy_levels[i].second;
      }
    }

    MPI_Bcast(level_energies.data(), num_levels, MPI_DOUBLE, 0, world);
    MPI_Bcast(btypes.data(), num_levels, MPI_INT, 0, world);

    // non-root processors unpack into energy_levels
    if (me != 0) {
      for (int i = 0; i < num_levels; i++) {
        energy_levels[i] = std::make_pair(level_energies[i], btypes[i]);
      }
    }
  }

  // broadcast complementarity map
  int map_size = complementarity_map.size();
  MPI_Bcast(&map_size, 1, MPI_INT, 0, world);

  if (map_size > 0) {
    // pack map into flat arrays for MPI broadcast
    std::vector<tagint> tags1(map_size);
    std::vector<tagint> tags2(map_size);
    std::vector<double> all_energies(map_size * num_temperatures);

    if (me == 0) {
      int idx = 0;
      for (const auto &entry : complementarity_map) {
        tags1[idx] = entry.first.first;
        tags2[idx] = entry.first.second;
        for (int t = 0; t < num_temperatures; t++) {
          all_energies[idx * num_temperatures + t] = entry.second[t];
        }
        idx++;
      }
    }

    MPI_Bcast(tags1.data(), map_size, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(tags2.data(), map_size, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(all_energies.data(), map_size * num_temperatures, MPI_DOUBLE, 0, world);

    // non-root processors unpack into complementarity_map
    if (me != 0) {
      for (int i = 0; i < map_size; i++) {
        std::vector<double> energies(num_temperatures);
        for (int t = 0; t < num_temperatures; t++) {
          energies[t] = all_energies[i * num_temperatures + t];
        }
        complementarity_map[std::make_pair(tags1[i], tags2[i])] = energies;
      }
    }
  }

  // log info about loaded data
  if (me == 0) {
    if (screen)
      fprintf(screen,"Fix dnafold/bond/hyb: Read %d temperatures, %d bond types, and %d complementarity pairs from '%s'\n",
              num_temperatures, num_levels, map_size, complementarity_file);
    if (logfile)
      fprintf(logfile,"Fix dnafold/bond/hyb: Read %d temperatures, %d bond types, and %d complementarity pairs from '%s'\n",
              num_temperatures, num_levels, map_size, complementarity_file);
  }
}

/* ----------------------------------------------------------------------
   Get temperature-interpolated binding energy for an atom pair.
   Returns -BIG if pair is not in complementarity map (not complementary).
------------------------------------------------------------------------- */

double FixDnafoldBondHyb::get_interpolated_energy(tagint tag_i, tagint tag_j)
{
  // ensure tag1 < tag2 for consistent map lookup
  tagint tag1 = (tag_i < tag_j) ? tag_i : tag_j;
  tagint tag2 = (tag_i < tag_j) ? tag_j : tag_i;

  // lookup pair in complementarity map
  auto it = complementarity_map.find(std::make_pair(tag1, tag2));
  if (it == complementarity_map.end()) return -BIG;  // not complementary

  const std::vector<double>& energies = it->second;

  // get current temperature from the LAMMPS variable
  double T = input->variable->compute_equal(tvar_index);

  // verify temperature is within the valid range from file
  if (T < temperatures.front() || T > temperatures.back()) {
    error->all(FLERR, fmt::format(
      "Fix dnafold/bond/hyb: Temperature {} outside complementarity file range [{}, {}]",
      T, temperatures.front(), temperatures.back()));
  }

  // perform linear interpolation between bracketing temperature points
  for (int i = 0; i < num_temperatures - 1; i++) {
    if (T >= temperatures[i] && T <= temperatures[i+1]) {
      double t0 = temperatures[i];
      double t1 = temperatures[i+1];
      double e0 = energies[i];
      double e1 = energies[i+1];
      double frac = (T - t0) / (t1 - t0);
      return e0 + frac * (e1 - e0);
    }
  }

  return energies.back();  // should not reach here
}

/* ----------------------------------------------------------------------
   Get the appropriate bond type for an atom pair based on their
   interpolated binding energy. Returns 0 if energy is below threshold.
------------------------------------------------------------------------- */

int FixDnafoldBondHyb::get_bond_type(tagint tag_i, tagint tag_j)
{
  double target_energy = get_interpolated_energy(tag_i, tag_j);

  // not complementary - return 0 (no bond)
  if (target_energy < -BIG/2) return 0;

  // if target_energy < lowest available energy, return 0 (no bond)
  // energy_levels is sorted descending, so .back() is the lowest (weakest)
  if (target_energy < energy_levels.back().first) {
    return 0;
  }

  // find the bond type with energy closest to target
  int best_btype = energy_levels[0].second;
  double best_diff = fabs(target_energy - energy_levels[0].first);

  for (size_t i = 1; i < energy_levels.size(); i++) {
    double diff = fabs(target_energy - energy_levels[i].first);
    if (diff < best_diff) {
      best_diff = diff;
      best_btype = energy_levels[i].second;
    }
  }

  return best_btype;
}

/* ----------------------------------------------------------------------
   Get the energy depth for an atom pair (for partner selection).
   Returns BIG if not complementary (worst possible for comparison).
------------------------------------------------------------------------- */

double FixDnafoldBondHyb::get_energy_depth(tagint tag_i, tagint tag_j)
{
  double energy = get_interpolated_energy(tag_i, tag_j);

  // not complementary - return very large value (for "worst" comparison)
  if (energy < -BIG/2) return BIG;

  return energy;
}

/* ----------------------------------------------------------------------
   Main function called every nevery timesteps.
   Two passes:
   (1) Check existing hyb bonds - downgrade if stretched or energy dropped
   (2) Upgrade dummy bonds to hyb bonds for close complementary pairs
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::post_integrate()
{
  int i,j,m;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int itype, jtype, btype_ij;
  tagint itag, jtag;

  if (update->ntimestep % nevery) return;

  int *hyb_status = atom->ivector[hyb_status_index];
  int *size = atom->ivector[size_index];

  // acquire updated ghost atom positions and properties
  comm->forward_comm();

  // resize partner arrays if atom count increased
  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(final_partner);
    memory->destroy(partner_bond_type);
    memory->destroy(dist_sq);
    memory->destroy(partner_energy);
    nmax = atom->nmax;
    memory->create(partner, nmax, "dnafold/bond/hyb:partner");
    memory->create(final_partner, nmax, "dnafold/bond/hyb:final_partner");
    memory->create(partner_bond_type, nmax, "dnafold/bond/hyb:partner_bond_type");
    memory->create(dist_sq, nmax, "dnafold/bond/hyb:dist_sq");
    memory->create(partner_energy, nmax, "dnafold/bond/hyb:partner_energy");
  }

  // get atom data pointers
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;

  // initialize partner arrays for all atoms (local + ghost)
  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    final_partner[i] = 0;
    partner_bond_type[i] = 0;
    dist_sq[i] = BIG;
    partner_energy[i] = BIG;
  }

  // ========== FIRST PASS: Check existing hyb bonds ==========
  // Downgrade to dummy if:
  // - Distance exceeds cutoff
  // - Energy dropped below threshold at current temperature
  // Update bond type if energy changed to different level

  int local_downgrade_count = 0;
  int update_count = 0;
  std::vector<tagint> downgraded_type1_tags;  // track type 1 atoms for angle removal
  std::vector<std::pair<tagint, int>> hyb_status_changes;  // track (tag, delta) for ghost atoms

  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    // skip atoms with no hybridized bonds
    if (hyb_status[i] == 0) continue;

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop through i's bonds looking for hyb bonds
    int ib = 0;
    while (ib < num_bond[i]) {
      // skip if not a hyb bond type (skip dummy bonds, backbone bonds, etc.)
      if (!is_hyb_bond_type(bond_type[i][ib])) {
        ib++;
        continue;
      }

      // this is a hyb bond - get partner info
      jtag = bond_atom[i][ib];
      j = atom->map(jtag);
      if (j < 0) {
        error->one(FLERR,"Fix dnafold/bond/hyb: Bonded atom not found in ghost atoms. "
                         "Increase communication cutoff with 'comm_modify cutoff'");
      }

      // calculate distance with minimum image convention
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      if (domain->xperiodic) {
        if (delx > domain->xprd_half) delx -= domain->xprd;
        else if (delx < -domain->xprd_half) delx += domain->xprd;
      }
      if (domain->yperiodic) {
        if (dely > domain->yprd_half) dely -= domain->yprd;
        else if (dely < -domain->yprd_half) dely += domain->yprd;
      }
      if (domain->zperiodic) {
        if (delz > domain->zprd_half) delz -= domain->zprd;
        else if (delz < -domain->zprd_half) delz += domain->zprd;
      }
      rsq = delx*delx + dely*dely + delz*delz;

      // get the correct bond type at current temperature
      int correct_btype = get_bond_type(itag, jtag);

      // determine action: downgrade to dummy, update bond type, or keep as-is
      bool should_downgrade = false;

      if (rsq > cutoff_sq) {
        // too far - downgrade to dummy
        should_downgrade = true;
      } else if (correct_btype == 0) {
        // energy dropped below threshold at current T - downgrade to dummy
        should_downgrade = true;
      }

      if (should_downgrade) {
        // downgrade to dummy bond
        bond_type[i][ib] = dummy_bond_type;

        // calculate min_size and subtract from hyb_status for both atoms
        int min_size = (size[i] < size[j]) ? size[i] : size[j];
        hyb_status[i] -= min_size;

        // update j's hyb_status - track for broadcast if j is a ghost
        if (j < nlocal) {
          hyb_status[j] -= min_size;
        } else {
          hyb_status_changes.push_back(std::make_pair(jtag, -min_size));
        }

        // if i is type 1, mark for angle removal
        if (itype == iatomtype) {
          downgraded_type1_tags.push_back(itag);
        }

        local_downgrade_count++;
      } else if (bond_type[i][ib] != correct_btype) {
        // bond type changed due to temperature - update it
        bond_type[i][ib] = correct_btype;
        update_count++;
        // hyb_status doesn't change since it's still a hyb bond
      }

      ib++;
    }
  }

  // ========== Remove angles involving downgraded type 1 atoms ==========
  // Gather all downgraded type 1 tags across processors

  int nlocal_downgraded = downgraded_type1_tags.size();
  int ntotal_downgraded = 0;
  MPI_Allreduce(&nlocal_downgraded, &ntotal_downgraded, 1, MPI_INT, MPI_SUM, world);

  std::vector<tagint> all_downgraded_type1_tags;
  if (ntotal_downgraded > 0) {
    // gather counts from all processors
    std::vector<int> recvcounts(nprocs);
    MPI_Allgather(&nlocal_downgraded, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

    // calculate displacements for allgatherv
    std::vector<int> displs(nprocs);
    displs[0] = 0;
    for (int p = 1; p < nprocs; p++) {
      displs[p] = displs[p-1] + recvcounts[p-1];
    }

    // gather all tags
    all_downgraded_type1_tags.resize(ntotal_downgraded);
    MPI_Allgatherv(downgraded_type1_tags.data(), nlocal_downgraded, MPI_LMP_TAGINT,
                   all_downgraded_type1_tags.data(), recvcounts.data(), displs.data(),
                   MPI_LMP_TAGINT, world);
  }

  // remove angles involving any downgraded type 1 atoms
  if (ntotal_downgraded > 0) {
    int **angle_type = atom->angle_type;
    tagint **angle_atom1 = atom->angle_atom1;
    tagint **angle_atom2 = atom->angle_atom2;
    tagint **angle_atom3 = atom->angle_atom3;
    int *num_angle = atom->num_angle;

    // loop through all local atoms' angles
    for (i = 0; i < nlocal; i++) {
      int ia = 0;
      while (ia < num_angle[i]) {
        bool should_remove = false;

        // check if any of the three atoms in this angle match a downgraded type 1 atom
        for (tagint downgraded_tag : all_downgraded_type1_tags) {
          if (angle_atom1[i][ia] == downgraded_tag ||
              angle_atom2[i][ia] == downgraded_tag ||
              angle_atom3[i][ia] == downgraded_tag) {
            should_remove = true;
            break;
          }
        }

        if (should_remove) {
          // remove angle by shifting remaining angles down
          for (int k = ia; k < num_angle[i] - 1; k++) {
            angle_type[i][k] = angle_type[i][k+1];
            angle_atom1[i][k] = angle_atom1[i][k+1];
            angle_atom2[i][k] = angle_atom2[i][k+1];
            angle_atom3[i][k] = angle_atom3[i][k+1];
          }
          num_angle[i]--;
          // don't increment ia since we shifted
        } else {
          ia++;
        }
      }
    }
  }

  // ========== SECOND PASS: Upgrade dummy bonds to hyb bonds ==========
  // Find the best partner (closest distance with strongest binding) for each atom

  for (i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;

    // skip if atom i already at full hybridization capacity
    if (hyb_status[i] >= MAX_HYB_CAPACITY) continue;

    // atom i must be type 1 or type 2
    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop through i's bonds looking for dummy bonds to upgrade
    for (int k = 0; k < num_bond[i]; k++) {
      // skip if not a dummy bond
      if (bond_type[i][k] != dummy_bond_type) continue;

      // get partner atom
      jtag = bond_atom[i][k];
      j = atom->map(jtag);

      if (j < 0) continue;  // partner not found

      if (!(mask[j] & groupbit)) continue;

      // skip if atom j already at full hybridization capacity
      if (hyb_status[j] >= MAX_HYB_CAPACITY) continue;

      // atom j must be the opposite type from i
      jtype = type[j];
      if (itype == iatomtype && jtype != jatomtype) continue;
      if (itype == jatomtype && jtype != iatomtype) continue;

      // calculate distance with minimum image convention
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      if (domain->xperiodic) {
        if (delx > domain->xprd_half) delx -= domain->xprd;
        else if (delx < -domain->xprd_half) delx += domain->xprd;
      }
      if (domain->yperiodic) {
        if (dely > domain->yprd_half) dely -= domain->yprd;
        else if (dely < -domain->yprd_half) dely += domain->yprd;
      }
      if (domain->zperiodic) {
        if (delz > domain->zprd_half) delz -= domain->zprd;
        else if (delz < -domain->zprd_half) delz += domain->zprd;
      }
      rsq = delx*delx + dely*dely + delz*delz;

      // skip if beyond cutoff distance
      if (rsq > cutoff_sq) continue;

      // check if this pair is complementary and get bond type
      btype_ij = get_bond_type(itag, jtag);
      if (btype_ij == 0) continue;  // not complementary

      double energy_ij = get_energy_depth(itag, jtag);

      // check hybridization capacity constraint
      int min_size = (size[i] < size[j]) ? size[i] : size[j];
      if (hyb_status[i] + min_size > MAX_HYB_CAPACITY) continue;
      if (hyb_status[j] + min_size > MAX_HYB_CAPACITY) continue;

      // update partner for atom i if this is better
      // better = lower energy depth (stronger bond), or same energy but closer distance
      bool better_for_i = (energy_ij < partner_energy[i]) ||
                          (energy_ij == partner_energy[i] && rsq < dist_sq[i]);
      if (better_for_i) {
        partner[i] = jtag;
        partner_bond_type[i] = btype_ij;
        dist_sq[i] = rsq;
        partner_energy[i] = energy_ij;
      }

      // update partner for atom j if this is better
      // this is safe even if j is a ghost - we'll communicate this back
      bool better_for_j = (energy_ij < partner_energy[j]) ||
                          (energy_ij == partner_energy[j] && rsq < dist_sq[j]);
      if (better_for_j) {
        partner[j] = itag;
        partner_bond_type[j] = btype_ij;
        dist_sq[j] = rsq;
        partner_energy[j] = energy_ij;
      }
    }
  }

  // reverse comm: send ghost atom partner data back to home processors
  // home processor keeps the best (closest/strongest) partner
  commflag = 1;
  if (force->newton_pair) comm->reverse_comm(this);

  // forward comm: send finalized partner data to ghosts
  // ensures all atoms know about confirmed partners
  commflag = 1;
  comm->forward_comm(this);

  // ========== Create bonds for mutual partners ==========
  // Only create bond if both atoms chose each other as partners

  int local_create_count = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;

    // map partner tag to local or ghost index
    j = atom->map(partner[i]);
    if (j < 0) {
      // partner migrated out of ghost range between selection and creation
      // this is rare but possible - skip this pair (mutual check would fail anyway)
      partner[i] = 0;
      continue;
    }

    // both atoms must agree on being partners (mutual selection)
    if (partner[j] != tag[i]) continue;

    // with newton_bond on, only store bond on lower-tagged atom to avoid duplicates
    if (tag[i] > tag[j]) continue;

    // Re-validate types before bond creation
    int itype_check = type[i];
    int jtype_check = type[j];
    if (!((itype_check == iatomtype && jtype_check == jatomtype) ||
          (itype_check == jatomtype && jtype_check == iatomtype))) {
      continue;
    }

    // Re-validate capacity before creating bond
    int min_size_check = (size[i] < size[j]) ? size[i] : size[j];
    if (hyb_status[i] + min_size_check > MAX_HYB_CAPACITY) continue;
    if (hyb_status[j] + min_size_check > MAX_HYB_CAPACITY) continue;

    if (num_bond[i] >= atom->bond_per_atom)
      error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/hyb");

    // create the hybridization bond with appropriate bond type
    bond_type[i][num_bond[i]] = partner_bond_type[i];
    bond_atom[i][num_bond[i]] = tag[j];
    num_bond[i]++;

    // remove the dummy bond between i and j (it's being replaced)
    remove_dummy_bond(i, j);

    // calculate min_size and add to hyb_status for both atoms
    int min_size = (size[i] < size[j]) ? size[i] : size[j];
    hyb_status[i] += min_size;

    // update j's hyb_status - track for broadcast if j is a ghost
    if (j < nlocal) {
      hyb_status[j] += min_size;
    } else {
      hyb_status_changes.push_back(std::make_pair(tag[j], min_size));
    }

    // store final partners for bookkeeping
    final_partner[i] = tag[j];

    local_create_count++;
  }

  // ========== Broadcast hyb_status changes for ghost atoms ==========
  // When we modify hyb_status for a ghost atom, the real atom on its home
  // processor doesn't see the change. Gather all changes and apply them.

  int nlocal_changes = hyb_status_changes.size();
  int ntotal_changes = 0;
  MPI_Allreduce(&nlocal_changes, &ntotal_changes, 1, MPI_INT, MPI_SUM, world);

  if (ntotal_changes > 0) {
    // gather counts from all processors
    std::vector<int> recvcounts(nprocs);
    MPI_Allgather(&nlocal_changes, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

    // calculate displacements for allgatherv
    std::vector<int> displs(nprocs);
    displs[0] = 0;
    for (int p = 1; p < nprocs; p++) {
      displs[p] = displs[p-1] + recvcounts[p-1];
    }

    // pack local changes into flat arrays
    std::vector<tagint> local_tags(nlocal_changes);
    std::vector<int> local_deltas(nlocal_changes);
    for (int c = 0; c < nlocal_changes; c++) {
      local_tags[c] = hyb_status_changes[c].first;
      local_deltas[c] = hyb_status_changes[c].second;
    }

    // gather all changes from all processors
    std::vector<tagint> all_tags(ntotal_changes);
    std::vector<int> all_deltas(ntotal_changes);

    MPI_Allgatherv(local_tags.data(), nlocal_changes, MPI_LMP_TAGINT,
                   all_tags.data(), recvcounts.data(), displs.data(),
                   MPI_LMP_TAGINT, world);
    MPI_Allgatherv(local_deltas.data(), nlocal_changes, MPI_INT,
                   all_deltas.data(), recvcounts.data(), displs.data(),
                   MPI_INT, world);

    // apply changes to local atoms owned by this processor
    for (int c = 0; c < ntotal_changes; c++) {
      int idx = atom->map(all_tags[c]);
      if (idx >= 0 && idx < nlocal) {
        hyb_status[idx] += all_deltas[c];
      }
    }
  }

  // ========== Finalize: aggregate counts and rebuild special lists ==========

  int create_count_all, downgrade_count_all, update_count_all;
  MPI_Allreduce(&local_create_count,&create_count_all,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&local_downgrade_count,&downgrade_count_all,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&update_count,&update_count_all,1,MPI_INT,MPI_SUM,world);
  create_count_total += create_count_all;
  downgrade_count_total += downgrade_count_all;
  create_count = create_count_all;
  downgrade_count = downgrade_count_all;

  // if any bonds changed, trigger reneighboring
  if (create_count || downgrade_count || update_count_all) {
    next_reneighbor = update->ntimestep;
  }
}

/* ----------------------------------------------------------------------
   Pack partner data for forward communication to ghost atoms.
------------------------------------------------------------------------- */

int FixDnafoldBondHyb::pack_forward_comm(int n, int *list, double *buf,
                                          int /* pbc_flag */, int * /* pbc */)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(partner[j]).d;
    buf[m++] = ubuf(partner_bond_type[j]).d;
    buf[m++] = dist_sq[j];
    buf[m++] = partner_energy[j];
  }
  return m;
}

/* ----------------------------------------------------------------------
   Unpack partner data from forward communication to ghost atoms.
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    partner[i] = (tagint) ubuf(buf[m++]).i;
    partner_bond_type[i] = (int) ubuf(buf[m++]).i;
    dist_sq[i] = buf[m++];
    partner_energy[i] = buf[m++];
  }
}

/* ----------------------------------------------------------------------
   Pack partner data for reverse communication from ghost to home processors.
------------------------------------------------------------------------- */

int FixDnafoldBondHyb::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = ubuf(partner_bond_type[i]).d;
    buf[m++] = dist_sq[i];
    buf[m++] = partner_energy[i];
  }
  return m;
}

/* ----------------------------------------------------------------------
   Unpack partner data from reverse communication.
   Home processor keeps the better partner (stronger bond or closer distance).
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];

    // compare incoming partner with current partner
    // keep the better one: lower energy depth (stronger) or closer distance
    double incoming_energy = buf[m+3];
    double incoming_dist_sq = buf[m+2];

    bool better = (incoming_energy < partner_energy[j]) ||
                  (incoming_energy == partner_energy[j] && incoming_dist_sq < dist_sq[j]);

    if (better) {
      partner[j] = (tagint) ubuf(buf[m++]).i;
      partner_bond_type[j] = (int) ubuf(buf[m++]).i;
      dist_sq[j] = buf[m++];
      partner_energy[j] = buf[m++];
    } else m += 4;
  }
}

/* ----------------------------------------------------------------------
   Remove a dummy bond between atoms i and j.
   Called when upgrading dummy bond to hybridization bond.
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::remove_dummy_bond(int i, int j)
{
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  tagint jtag = atom->tag[j];

  // look through i's bonds to find dummy bond to j
  for (int k = 0; k < num_bond[i]; k++) {
    if (bond_atom[i][k] == jtag && bond_type[i][k] == dummy_bond_type) {
      // found dummy bond - remove it by shifting remaining bonds down
      for (int m = k; m < num_bond[i] - 1; m++) {
        bond_type[i][m] = bond_type[i][m+1];
        bond_atom[i][m] = bond_atom[i][m+1];
      }
      num_bond[i]--;
      return;
    }
  }
}

/* ----------------------------------------------------------------------
   Check if a bond type is a hybridization bond type (from energy_levels).
   Returns true if the bond type is in the TYPES section of the complementarity file.
------------------------------------------------------------------------- */

bool FixDnafoldBondHyb::is_hyb_bond_type(int btype)
{
  for (const auto &level : energy_levels) {
    if (level.second == btype) return true;
  }
  return false;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondHyb::compute_vector(int n)
{
  // output vector: [0]=created, [1]=total_created, [2]=downgraded, [3]=total_downgraded
  if (n == 0) return (double) create_count;
  if (n == 1) return (double) create_count_total;
  if (n == 2) return (double) downgrade_count;
  return (double) downgrade_count_total;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondHyb::memory_usage()
{
  // estimate memory for partner arrays
  double bytes = (double)nmax * 2 * sizeof(tagint);  // partner, final_partner
  bytes += (double)nmax * sizeof(int);                // partner_bond_type
  bytes += (double)nmax * 2 * sizeof(double);         // dist_sq, partner_energy

  // estimate memory for complementarity map
  bytes += complementarity_map.size() * (2 * sizeof(tagint) + sizeof(int) + 32);

  return bytes;
}
