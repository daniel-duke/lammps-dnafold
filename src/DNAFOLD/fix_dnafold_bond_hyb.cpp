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
   DNAFOLD package: Mesoscopic DNA origami folding simulation

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

/* ---------------------------------------------------------------------- */

FixDnafoldBondHyb::FixDnafoldBondHyb(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), complementarity_file(nullptr), tvar(nullptr),
  partner(nullptr), final_partner(nullptr), partner_bond_type(nullptr),
  dist_sq(nullptr), partner_energy(nullptr)
{
  // syntax: fix ID group dnafold/bond/hyb nevery cutoff bond_type comp_file temp_var
  if (narg != 8) error->all(FLERR,"Illegal command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // parse nevery - how often to check for bond upgrades/downgrades
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal nevery");

  // parse cutoff - distance for hybridization
  cutoff_sq = utils::numeric(FLERR,arg[4],false,lmp);
  if (cutoff_sq <= 0.0) error->all(FLERR,"Illegal cutoff");
  cutoff_sq = cutoff_sq * cutoff_sq;

  // parse bond_type - dummy bond type to consider upgrading to hyb bonds
  dummy_bond_type = utils::inumeric(FLERR,arg[5],false,lmp);
  if (dummy_bond_type <= 0) error->all(FLERR,"Illegal dummy bond type");

  // parse comp_file - complementarity file path
  complementarity_file = utils::strdup(arg[6]);

  // parse temp_var - temperature variable name (must start with v_)
  if (strncmp(arg[7], "v_", 2) != 0)
    error->all(FLERR,"Temperature variable must start with v_");
  tvar = utils::strdup(arg[7] + 2);
  tvar_index = -1;

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
  // find the hyb_status_5p and hyb_status_3p custom properties
  int flag_hyb, cols_hyb;
  hyb_status_5p_index = atom->find_custom("hyb_status_5p", flag_hyb, cols_hyb);
  if (hyb_status_5p_index < 0)
    error->all(FLERR,"Could not find property 'i_hyb_status_5p'");
  if (flag_hyb != 0)
    error->all(FLERR,"Property 'i_hyb_status_5p' must be integer");

  hyb_status_3p_index = atom->find_custom("hyb_status_3p", flag_hyb, cols_hyb);
  if (hyb_status_3p_index < 0)
    error->all(FLERR,"Could not find property 'i_hyb_status_3p'");
  if (flag_hyb != 0)
    error->all(FLERR,"Property 'i_hyb_status_3p' must be integer");

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
   - TYPES: bond type to energy depth mapping
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
      // format: tag1 tag2 half_i half_j <ignored> energy1 energy2 ...
      if (current_section == PAIRS) {
        std::istringstream iss(line);
        tagint tag1, tag2;
        std::string ignored;
        int half_i_val, half_j_val;

        if (!(iss >> tag1 >> tag2 >> half_i_val >> half_j_val >> ignored)) {
          error->one(FLERR,fmt::format("Invalid PAIRS format in '{}' at line {}",
                                       complementarity_file, line_num));
        }

        if (half_i_val < 0 || half_i_val > 2 || half_j_val < 0 || half_j_val > 2) {
          error->one(FLERR,fmt::format("PAIRS line {} in '{}': half values must be 0, 1, or 2",
                                       line_num, complementarity_file));
        }

        // read energy values for each temperature point
        std::vector<double> energies;
        double energy;
        while (iss >> energy) {
          energies.push_back(energy);
        }

        // check the number of energy values
        if ((int)energies.size() != num_temperatures) {
          error->one(FLERR,fmt::format("PAIRS line {} in '{}' has {} energies, expected {}",
                                       line_num, complementarity_file,
                                       energies.size(), num_temperatures));
        }

        // store with tag1 < tag2 for consistent lookup
        if (tag1 > tag2) {
          std::swap(tag1, tag2);
          std::swap(half_i_val, half_j_val);
        }
        PairData pd;
        pd.half_i = half_i_val;
        pd.half_j = half_j_val;
        pd.energies = energies;
        complementarity_map[std::make_pair(tag1, tag2)] = pd;

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
          sorted_energies[i] = entry.second.energies[sort_indices[i]];
        }
        entry.second.energies = std::move(sorted_energies);
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

  // broadcast energy levels size
  int num_levels = energy_levels.size();
  MPI_Bcast(&num_levels, 1, MPI_INT, 0, world);

  // broadcast energy levels by packing into arrays
  if (num_levels > 0) {
    if (me != 0) energy_levels.resize(num_levels);
    std::vector<double> energies(num_levels);
    std::vector<int> btypes(num_levels);

    // if rank 0, pack data into flat arrays for MPI broadcast
    if (me == 0) {
      for (int i = 0; i < num_levels; i++) {
        energies[i] = energy_levels[i].first;
        btypes[i] = energy_levels[i].second;
      }
    }

    MPI_Bcast(energies.data(), num_levels, MPI_DOUBLE, 0, world);
    MPI_Bcast(btypes.data(), num_levels, MPI_INT, 0, world);

    // if non-root (not rank 0), unpack arrays into local variables
    if (me != 0) {
      for (int i = 0; i < num_levels; i++) {
        energy_levels[i] = std::make_pair(energies[i], btypes[i]);
      }
    }
  }

  // broadcast complementarity map size
  int map_size = complementarity_map.size();
  MPI_Bcast(&map_size, 1, MPI_INT, 0, world);

  // broadcast complementarity map by packing into arrays
  if (map_size > 0) {
    std::vector<tagint> tags1(map_size);
    std::vector<tagint> tags2(map_size);
    std::vector<int> halves_i(map_size);
    std::vector<int> halves_j(map_size);
    std::vector<double> all_energies(map_size * num_temperatures);

    // if rank 0, pack data into flat arrays for MPI broadcast
    if (me == 0) {
      int idx = 0;
      for (const auto &entry : complementarity_map) {
        tags1[idx] = entry.first.first;
        tags2[idx] = entry.first.second;
        halves_i[idx] = entry.second.half_i;
        halves_j[idx] = entry.second.half_j;
        for (int t = 0; t < num_temperatures; t++) {
          all_energies[idx * num_temperatures + t] = entry.second.energies[t];
        }
        idx++;
      }
    }

    MPI_Bcast(tags1.data(), map_size, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(tags2.data(), map_size, MPI_LMP_TAGINT, 0, world);
    MPI_Bcast(halves_i.data(), map_size, MPI_INT, 0, world);
    MPI_Bcast(halves_j.data(), map_size, MPI_INT, 0, world);
    MPI_Bcast(all_energies.data(), map_size * num_temperatures, MPI_DOUBLE, 0, world);

    // if non-root (not rank 0), unpack arrays into local variables
    if (me != 0) {
      for (int i = 0; i < map_size; i++) {
        PairData pd;
        pd.half_i = halves_i[i];
        pd.half_j = halves_j[i];
        pd.energies.resize(num_temperatures);
        for (int t = 0; t < num_temperatures; t++) {
          pd.energies[t] = all_energies[i * num_temperatures + t];
        }
        complementarity_map[std::make_pair(tags1[i], tags2[i])] = pd;
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

  const std::vector<double>& energies = it->second.energies;

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
   Main function called every nevery timesteps.
   Delegates to downgrade_bonds() and upgrade_bonds(), broadcasts
   hyb_status changes, then aggregates counts and triggers reneighboring.
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::post_integrate()
{
  if (update->ntimestep % nevery) return;

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

  // initialize partner arrays for all atoms (local + ghost)
  int nall = atom->nlocal + atom->nghost;
  for (int i = 0; i < nall; i++) {
    partner[i] = 0;
    final_partner[i] = 0;
    partner_bond_type[i] = 0;
    dist_sq[i] = BIG;
    partner_energy[i] = BIG;
  }

  // clear shared state for this timestep
  hyb_status_changes.clear();
  create_count = 0;
  downgrade_count = 0;
  update_count = 0;

  // phase 1: downgrade hyb bonds that no longer meet criteria, remove associated angles
  downgrade_bonds();

  // phase 2: upgrade dummy bonds to hyb bonds for close complementary pairs
  upgrade_bonds();

  // broadcast hyb_status changes (from both phases) to ghost atoms' home processors
  broadcast_hyb_status_changes();

  // accumulate counts across all MPI processors
  int create_count_all, downgrade_count_all, update_count_all;
  MPI_Allreduce(&create_count, &create_count_all, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&downgrade_count, &downgrade_count_all, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&update_count, &update_count_all, 1, MPI_INT, MPI_SUM, world);
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
   Downgrade hyb bonds that no longer meet validity criteria.
   A hyb bond is downgraded to dummy if:
   - Distance exceeds cutoff
   - Energy dropped below threshold at current temperature
   Also updates bond type in-place if energy changed to a different level.
   Removes angles associated with any downgraded type 1 atoms.
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::downgrade_bonds()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int *hyb_status_5p = atom->ivector[hyb_status_5p_index];
  int *hyb_status_3p = atom->ivector[hyb_status_3p_index];

  std::vector<tagint> downgraded_type1_tags;  // type 1 atoms with downgraded bonds

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    int itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    // skip atoms with no hybridized bonds on either side
    if (hyb_status_5p[i] == 0 && hyb_status_3p[i] == 0) continue;

    tagint itag = tag[i];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];

    // loop through i's bonds looking for hyb bonds
    int ib = 0;
    while (ib < num_bond[i]) {
      // skip if not a hyb bond type
      if (!is_hyb_bond_type(bond_type[i][ib])) {
        ib++;
        continue;
      }

      // this is a hyb bond - get partner info
      tagint jtag = bond_atom[i][ib];
      int j = atom->map(jtag);
      if (j < 0) {
        error->one(FLERR,"Fix dnafold/bond/hyb: Bonded atom not found in ghost atoms. "
                         "Increase communication cutoff with 'comm_modify cutoff'");
      }

      // calculate distance with minimum image convention
      double delx = xtmp - x[j][0];
      double dely = ytmp - x[j][1];
      double delz = ztmp - x[j][2];
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
      double rsq = delx*delx + dely*dely + delz*delz;

      // get the correct bond type at current temperature
      int correct_btype = get_bond_type(itag, jtag);

      // determine action: downgrade to dummy, update bond type, or keep as-is
      bool should_downgrade = (rsq > cutoff_sq) || (correct_btype == 0);

      if (should_downgrade) {
        // downgrade to dummy bond
        bond_type[i][ib] = dummy_bond_type;

        // look up which halves this bond occupied for each atom
        std::pair<int,int> halves = get_halves(itag, jtag);
        int half_for_i = halves.first;
        int half_for_j = halves.second;

        // clear the occupied sides for atom i
        if (half_for_i == 0 || half_for_i == 1) hyb_status_5p[i] = 0;
        if (half_for_i == 0 || half_for_i == 2) hyb_status_3p[i] = 0;

        // clear the occupied sides for atom j (track for broadcast if ghost)
        int d5p_j = (half_for_j == 0 || half_for_j == 1) ? -1 : 0;
        int d3p_j = (half_for_j == 0 || half_for_j == 2) ? -1 : 0;
        if (j < nlocal) {
          hyb_status_5p[j] += d5p_j;
          hyb_status_3p[j] += d3p_j;
        } else {
          HybStatusChange chg;
          chg.tag = jtag; chg.delta_5p = d5p_j; chg.delta_3p = d3p_j;
          hyb_status_changes.push_back(chg);
        }

        // if i is type 1, mark for angle removal
        if (itype == iatomtype) {
          downgraded_type1_tags.push_back(itag);
        }

        downgrade_count++;
      } else if (bond_type[i][ib] != correct_btype) {
        // bond type changed due to temperature - update in-place
        bond_type[i][ib] = correct_btype;
        update_count++;
      }

      ib++;
    }
  }

  // === Remove angles involving downgraded type 1 atoms ===
  // Gather downgraded type 1 tags across all processors

  int nlocal_downgraded = (int)downgraded_type1_tags.size();
  int ntotal_downgraded = 0;
  MPI_Allreduce(&nlocal_downgraded, &ntotal_downgraded, 1, MPI_INT, MPI_SUM, world);

  std::vector<tagint> all_downgraded_type1_tags;
  if (ntotal_downgraded > 0) {
    std::vector<int> recvcounts(nprocs);
    MPI_Allgather(&nlocal_downgraded, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

    std::vector<int> displs(nprocs);
    displs[0] = 0;
    for (int p = 1; p < nprocs; p++) {
      displs[p] = displs[p-1] + recvcounts[p-1];
    }

    all_downgraded_type1_tags.resize(ntotal_downgraded);
    MPI_Allgatherv(downgraded_type1_tags.data(), nlocal_downgraded, MPI_LMP_TAGINT,
                   all_downgraded_type1_tags.data(), recvcounts.data(), displs.data(),
                   MPI_LMP_TAGINT, world);
  }

  // remove angles where any atom matches a downgraded type 1 atom
  int local_angles_removed = 0;
  if (ntotal_downgraded > 0) {
    int **angle_type = atom->angle_type;
    tagint **angle_atom1 = atom->angle_atom1;
    tagint **angle_atom2 = atom->angle_atom2;
    tagint **angle_atom3 = atom->angle_atom3;
    int *num_angle = atom->num_angle;

    for (int i = 0; i < nlocal; i++) {
      int ia = 0;
      while (ia < num_angle[i]) {
        bool should_remove = false;

        for (tagint downgraded_tag : all_downgraded_type1_tags) {
          if (angle_atom1[i][ia] == downgraded_tag ||
              angle_atom2[i][ia] == downgraded_tag ||
              angle_atom3[i][ia] == downgraded_tag) {
            should_remove = true;
            break;
          }
        }

        if (should_remove) {
          for (int k = ia; k < num_angle[i] - 1; k++) {
            angle_type[i][k] = angle_type[i][k+1];
            angle_atom1[i][k] = angle_atom1[i][k+1];
            angle_atom2[i][k] = angle_atom2[i][k+1];
            angle_atom3[i][k] = angle_atom3[i][k+1];
          }
          num_angle[i]--;
          local_angles_removed++;
        } else {
          ia++;
        }
      }
    }
  }

  // update global angle counter to reflect removed angles
  int total_angles_removed = 0;
  MPI_Allreduce(&local_angles_removed, &total_angles_removed, 1, MPI_INT, MPI_SUM, world);
  atom->nangles -= total_angles_removed;
}

/* ----------------------------------------------------------------------
   Upgrade dummy bonds to hyb bonds for close complementary atom pairs.
   Uses mutual partner selection: each atom picks its best candidate,
   then reverse/forward comm resolves ghost selections, and bonds are
   only created when both atoms mutually agree on each other as partners.
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::upgrade_bonds()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int *hyb_status_5p = atom->ivector[hyb_status_5p_index];
  int *hyb_status_3p = atom->ivector[hyb_status_3p_index];

  // === Find best partner candidates via dummy bonds ===

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    // atom i must be type 1 or type 2
    int itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    tagint itag = tag[i];
    double xtmp = x[i][0];
    double ytmp = x[i][1];
    double ztmp = x[i][2];

    // loop through i's bonds looking for dummy bonds to upgrade
    for (int k = 0; k < num_bond[i]; k++) {
      if (bond_type[i][k] != dummy_bond_type) continue;

      tagint jtag = bond_atom[i][k];
      int j = atom->map(jtag);

      if (j < 0) continue;
      if (!(mask[j] & groupbit)) continue;

      // atom j must be the opposite type from i
      int jtype = type[j];
      if (itype == iatomtype && jtype != jatomtype) continue;
      if (itype == jatomtype && jtype != iatomtype) continue;

      // calculate distance with minimum image convention
      double delx = xtmp - x[j][0];
      double dely = ytmp - x[j][1];
      double delz = ztmp - x[j][2];
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
      double rsq = delx*delx + dely*dely + delz*delz;

      if (rsq > cutoff_sq) continue;

      int btype_ij = get_bond_type(itag, jtag);
      if (btype_ij == 0) continue;

      double energy_ij = get_interpolated_energy(itag, jtag);

      // check capacity based on which halves this bond would occupy
      std::pair<int,int> halves = get_halves(itag, jtag);
      int half_for_i = halves.first;
      int half_for_j = halves.second;

      bool i_has_room = (half_for_i == 0) ? (hyb_status_5p[i] == 0 && hyb_status_3p[i] == 0)
                      : (half_for_i == 1) ? (hyb_status_5p[i] == 0)
                                          : (hyb_status_3p[i] == 0);
      bool j_has_room = (half_for_j == 0) ? (hyb_status_5p[j] == 0 && hyb_status_3p[j] == 0)
                      : (half_for_j == 1) ? (hyb_status_5p[j] == 0)
                                          : (hyb_status_3p[j] == 0);
      if (!i_has_room || !j_has_room) continue;

      // update partner for atom i if this candidate is better
      // better = stronger bond (lower energy depth), or same strength but closer
      bool better_for_i = (energy_ij < partner_energy[i]) ||
                          (energy_ij == partner_energy[i] && rsq < dist_sq[i]);
      if (better_for_i) {
        partner[i] = jtag;
        partner_bond_type[i] = btype_ij;
        dist_sq[i] = rsq;
        partner_energy[i] = energy_ij;
      }

      // update partner for atom j if this candidate is better
      // safe even if j is ghost - reverse comm will reconcile to home processor
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

  // reverse comm: send ghost atom partner selections back to home processors
  // home processor keeps the best (strongest bond or closest distance)
  commflag = 1;
  if (force->newton_pair) comm->reverse_comm(this);

  // forward comm: send finalized partner data to ghost atoms
  // ensures all atoms know about confirmed partners before bond creation
  commflag = 1;
  comm->forward_comm(this);

  // === Create bonds for mutually selected partners ===

  for (int i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;

    int j = atom->map(partner[i]);
    if (j < 0) {
      // partner migrated out of ghost range - skip
      partner[i] = 0;
      continue;
    }

    // both atoms must have chosen each other (mutual selection)
    if (partner[j] != tag[i]) continue;

    // with newton_bond on, only store bond on lower-tagged atom
    if (tag[i] > tag[j]) continue;

    // re-validate types before bond creation (protect against data corruption)
    int itype_check = type[i];
    int jtype_check = type[j];
    if (!((itype_check == iatomtype && jtype_check == jatomtype) ||
          (itype_check == jatomtype && jtype_check == iatomtype))) {
      continue;
    }

    // re-validate capacity before creating bond (protect against data corruption)
    std::pair<int,int> halves_check = get_halves(tag[i], tag[j]);
    int half_for_i_check = halves_check.first;
    int half_for_j_check = halves_check.second;

    bool i_ok = (half_for_i_check == 0) ? (hyb_status_5p[i] == 0 && hyb_status_3p[i] == 0)
              : (half_for_i_check == 1) ? (hyb_status_5p[i] == 0)
                                        : (hyb_status_3p[i] == 0);
    bool j_ok = (half_for_j_check == 0) ? (hyb_status_5p[j] == 0 && hyb_status_3p[j] == 0)
              : (half_for_j_check == 1) ? (hyb_status_5p[j] == 0)
                                        : (hyb_status_3p[j] == 0);
    if (!i_ok || !j_ok) continue;

    if (num_bond[i] >= atom->bond_per_atom)
      error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/hyb");

    // create the hybridization bond (replaces the dummy bond)
    bond_type[i][num_bond[i]] = partner_bond_type[i];
    bond_atom[i][num_bond[i]] = tag[j];
    num_bond[i]++;
    remove_dummy_bond(i, j);

    // mark the occupied sides for atom i
    if (half_for_i_check == 0 || half_for_i_check == 1) hyb_status_5p[i] = 1;
    if (half_for_i_check == 0 || half_for_i_check == 2) hyb_status_3p[i] = 1;

    // mark the occupied sides for atom j (track for broadcast if ghost)
    int d5p_j = (half_for_j_check == 0 || half_for_j_check == 1) ? 1 : 0;
    int d3p_j = (half_for_j_check == 0 || half_for_j_check == 2) ? 1 : 0;
    if (j < nlocal) {
      hyb_status_5p[j] += d5p_j;
      hyb_status_3p[j] += d3p_j;
    } else {
      HybStatusChange chg;
      chg.tag = tag[j]; chg.delta_5p = d5p_j; chg.delta_3p = d3p_j;
      hyb_status_changes.push_back(chg);
    }

    final_partner[i] = tag[j];
    create_count++;
  }
}

/* ----------------------------------------------------------------------
   Broadcast hyb_status changes for ghost atoms to their home processors.
   Both downgrade_bonds() and upgrade_bonds() may modify hyb_status for
   ghost atoms; those changes are invisible to the ghost's home processor
   until this broadcast applies them.
------------------------------------------------------------------------- */

void FixDnafoldBondHyb::broadcast_hyb_status_changes()
{
  int *hyb_status_5p = atom->ivector[hyb_status_5p_index];
  int *hyb_status_3p = atom->ivector[hyb_status_3p_index];
  int nlocal = atom->nlocal;

  int nlocal_changes = (int)hyb_status_changes.size();
  int ntotal_changes = 0;
  MPI_Allreduce(&nlocal_changes, &ntotal_changes, 1, MPI_INT, MPI_SUM, world);

  if (ntotal_changes > 0) {
    std::vector<int> recvcounts(nprocs);
    MPI_Allgather(&nlocal_changes, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);

    std::vector<int> displs(nprocs);
    displs[0] = 0;
    for (int p = 1; p < nprocs; p++) {
      displs[p] = displs[p-1] + recvcounts[p-1];
    }

    // pack local changes into flat arrays
    std::vector<tagint> local_tags(nlocal_changes);
    std::vector<int> local_deltas_5p(nlocal_changes);
    std::vector<int> local_deltas_3p(nlocal_changes);
    for (int c = 0; c < nlocal_changes; c++) {
      local_tags[c]      = hyb_status_changes[c].tag;
      local_deltas_5p[c] = hyb_status_changes[c].delta_5p;
      local_deltas_3p[c] = hyb_status_changes[c].delta_3p;
    }

    // gather all changes from all processors
    std::vector<tagint> all_tags(ntotal_changes);
    std::vector<int> all_deltas_5p(ntotal_changes);
    std::vector<int> all_deltas_3p(ntotal_changes);

    MPI_Allgatherv(local_tags.data(), nlocal_changes, MPI_LMP_TAGINT,
                   all_tags.data(), recvcounts.data(), displs.data(),
                   MPI_LMP_TAGINT, world);
    MPI_Allgatherv(local_deltas_5p.data(), nlocal_changes, MPI_INT,
                   all_deltas_5p.data(), recvcounts.data(), displs.data(),
                   MPI_INT, world);
    MPI_Allgatherv(local_deltas_3p.data(), nlocal_changes, MPI_INT,
                   all_deltas_3p.data(), recvcounts.data(), displs.data(),
                   MPI_INT, world);

    // apply changes to atoms owned by this processor
    for (int c = 0; c < ntotal_changes; c++) {
      int idx = atom->map(all_tags[c]);
      if (idx >= 0 && idx < nlocal) {
        hyb_status_5p[idx] += all_deltas_5p[c];
        hyb_status_3p[idx] += all_deltas_3p[c];
      }
    }
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
   Return the half values (which side each atom contributes) for a pair.
   Returns {half for tag_i, half for tag_j}.
   Returns {-1, -1} if pair not in map.
------------------------------------------------------------------------- */

std::pair<int,int> FixDnafoldBondHyb::get_halves(tagint tag_i, tagint tag_j)
{
  tagint tag1 = (tag_i < tag_j) ? tag_i : tag_j;
  tagint tag2 = (tag_i < tag_j) ? tag_j : tag_i;

  auto it = complementarity_map.find(std::make_pair(tag1, tag2));
  if (it == complementarity_map.end()) return std::make_pair(-1, -1);

  // half_i in map corresponds to tag1 (lower-tagged atom)
  if (tag_i < tag_j) return std::make_pair(it->second.half_i, it->second.half_j);
  else               return std::make_pair(it->second.half_j, it->second.half_i);
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
  double bytes = (double)nmax * 2 * sizeof(tagint);   // partner, final_partner
  bytes += (double)nmax * sizeof(int);                // partner_bond_type
  bytes += (double)nmax * 2 * sizeof(double);         // dist_sq, partner_energy

  // estimate memory for complementarity map
  bytes += complementarity_map.size() * (2 * sizeof(tagint) + sizeof(int) + 32);

  return bytes;
}
