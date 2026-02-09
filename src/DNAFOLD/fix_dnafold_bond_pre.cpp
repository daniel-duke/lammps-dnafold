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

   fix dnafold/bond/pre creates and removes "dummy" bonds between
   complementary atom pairs. These dummy bonds act as precursors to
   hybridization bonds - they identify which pairs could potentially
   hybridize but don't yet contribute to the hybridization state.
------------------------------------------------------------------------- */

#include "fix_dnafold_bond_pre.h"

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

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double BIG = 1.0e20;
static constexpr int MAX_HYB_CAPACITY = 2;

/* ---------------------------------------------------------------------- */

FixDnafoldBondPre::FixDnafoldBondPre(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), complementarity_file(nullptr), tvar(nullptr)
{
  // syntax: fix ID group dnafold/bond/pre nevery cutoff dummy_btype compfile tempvar
  if (narg != 8) error->all(FLERR,"Illegal fix dnafold/bond/pre command");

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

  // parse nevery - how often to check for bond creation/removal
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/bond/pre command");

  // set up fix flags for reneighboring and output vector
  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  vector_flag = 1;
  size_vector = 4;
  global_freq = 1;
  extvector = 0;

  // dummy bonds form between type 1 (scaffold) and type 2 (staple) atoms only
  iatomtype = 1;
  jatomtype = 2;

  // parse cutoff distance for dummy bond creation
  cutoff_sq = utils::numeric(FLERR, arg[4], false, lmp);
  if (cutoff_sq <= 0.0) error->all(FLERR,"Illegal cutoff");
  cutoff_sq = cutoff_sq * cutoff_sq;

  // parse dummy bond type
  dummy_bond_type = utils::inumeric(FLERR, arg[5], false, lmp);
  if (dummy_bond_type <= 0) error->all(FLERR,"Illegal dummy bond type");

  // parse complementarity file path
  complementarity_file = utils::strdup(arg[6]);

  // parse temperature variable name (must start with v_)
  if (strncmp(arg[7], "v_", 2) != 0)
    error->all(FLERR,"Temperature variable for fix dnafold/bond/pre must start with v_");
  tvar = utils::strdup(arg[7] + 2);
  tvar_index = -1;

  // initialize counters and data structures
  num_temperatures = 0;
  min_energy_threshold = 0.0;

  create_count = 0;
  remove_count = 0;
  create_count_total = 0;
  remove_count_total = 0;

  // read complementarity data from file
  read_complementarity_file();
}

/* ---------------------------------------------------------------------- */

FixDnafoldBondPre::~FixDnafoldBondPre()
{
  delete[] complementarity_file;
  delete[] tvar;
}

/* ---------------------------------------------------------------------- */

int FixDnafoldBondPre::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondPre::init()
{
  // find the hyb_status custom property (tracks hybridization slots used)
  int flag_hyb, cols_hyb;
  hyb_status_index = atom->find_custom("hyb_status", flag_hyb, cols_hyb);
  if (hyb_status_index < 0)
    error->all(FLERR,"Could not find hyb_status property for fix dnafold/bond/pre");
  if (flag_hyb != 0)
    error->all(FLERR,"Property hyb_status must be integer");

  // find the size custom property (1 for half-beads, 2 for whole beads)
  int flag_size, cols_size;
  size_index = atom->find_custom("size", flag_size, cols_size);
  if (size_index < 0)
    error->all(FLERR,"Could not find size property for fix dnafold/bond/pre");
  if (flag_size != 0)
    error->all(FLERR,"Property size must be integer");

  // find and validate the temperature variable
  tvar_index = input->variable->find(tvar);
  if (tvar_index < 0)
    error->all(FLERR,"Variable {} for fix dnafold/bond/pre does not exist", tvar);
  if (!input->variable->equalstyle(tvar_index))
    error->all(FLERR,"Variable {} for fix dnafold/bond/pre must be equal-style", tvar);

  // request an occasional neighbor list for finding nearby atom pairs
  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);

  // validate system is molecular
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/bond/pre with non-molecular system");

  // validate bond style is defined
  if (force->bond == nullptr)
    error->all(FLERR,"Must define bond_style");

  // validate dummy bond type is within range
  if (dummy_bond_type > atom->nbondtypes)
    error->all(FLERR,"Invalid dummy bond type");

  // require newton bond on for proper bond storage
  if (force->newton_bond == 0)
    error->all(FLERR,"Fix dnafold/bond/pre requires newton bond on");
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondPre::init_list(int /* id */, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixDnafoldBondPre::setup(int /* vflag */)
{
  post_integrate();
}

/* ----------------------------------------------------------------------
   Read complementarity file containing temperature-dependent binding energies.
   File format has three sections:
   - TEMPERATURES: list of temperature values
   - TYPES: bond type to energy mapping (used to determine min threshold)
   - PAIRS: atom tag pairs with energies at each temperature
------------------------------------------------------------------------- */

void FixDnafoldBondPre::read_complementarity_file()
{
  std::vector<double> type_energies;

  // rank 0 reads the file, then broadcasts data to all processors
  if (me == 0) {
    std::ifstream file(complementarity_file);
    if (!file.is_open())
      error->one(FLERR, fmt::format("Cannot open complementarity file '{}'", complementarity_file));

    std::string line;
    int line_num = 0;
    enum Section { NONE, TEMPERATURES, TYPES, PAIRS };
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
      if (trimmed == "TEMPERATURES") {
        current_section = TEMPERATURES;
        continue;
      } else if (trimmed == "TYPES") {
        current_section = TYPES;
        continue;
      } else if (trimmed == "PAIRS") {
        current_section = PAIRS;
        if (temperatures.empty()) {
          error->one(FLERR, fmt::format("TEMPERATURES section must appear before PAIRS in '{}'",
                                        complementarity_file));
        }
        continue;
      }

      // parse TEMPERATURES section: read list of temperature values
      if (current_section == TEMPERATURES) {
        std::istringstream iss(line);
        double temp;
        while (iss >> temp) {
          temperatures.push_back(temp);
        }
        if (temperatures.empty()) {
          error->one(FLERR, fmt::format("Invalid TEMPERATURES format in '{}' at line {}",
                                        complementarity_file, line_num));
        }
        num_temperatures = temperatures.size();
        continue;
      }

      // parse TYPES section: read bond type and energy depth pairs
      // we only need the energy values to determine minimum threshold
      if (current_section == TYPES) {
        std::istringstream iss(line);
        int btype;
        double energy;
        if (!(iss >> btype >> energy)) {
          error->one(FLERR, fmt::format("Invalid TYPES format in '{}' at line {}",
                                        complementarity_file, line_num));
        }
        type_energies.push_back(energy);
        continue;
      }

      // parse PAIRS section: read complementary atom pairs with per-temperature energies
      // format: tag1 tag2 <ignored_column> energy1 energy2 ...
      if (current_section == PAIRS) {
        std::istringstream iss(line);
        tagint tag1, tag2;
        std::string ignored_column;
        if (!(iss >> tag1 >> tag2 >> ignored_column)) {
          error->one(FLERR, fmt::format("Invalid PAIRS format in '{}' at line {}",
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
          error->one(FLERR, fmt::format("PAIRS line {} in '{}' has {} energies, expected {}",
                                        line_num, complementarity_file,
                                        energies.size(), num_temperatures));
        }

        // store with tag1 < tag2 for consistent lookup
        if (tag1 > tag2) std::swap(tag1, tag2);
        complementarity_map[std::make_pair(tag1, tag2)] = energies;

      } else if (current_section == NONE) {
        error->one(FLERR, fmt::format("Line {} in '{}' appears before any section header",
                                      line_num, complementarity_file));
      }
    }

    file.close();

    // validate required sections were found
    if (temperatures.empty()) {
      error->one(FLERR, fmt::format("No TEMPERATURES section found in '{}'", complementarity_file));
    }

    if (type_energies.empty()) {
      error->one(FLERR, fmt::format("No TYPES section found in '{}'", complementarity_file));
    }

    // minimum energy threshold: pairs with interpolated energy below this are not complementary
    min_energy_threshold = *std::min_element(type_energies.begin(), type_energies.end());

    if (complementarity_map.empty()) {
      error->warning(FLERR, "Complementarity file '{}' contains no valid pairs", complementarity_file);
    }
  }

  // broadcast temperature data to all processors
  MPI_Bcast(&num_temperatures, 1, MPI_INT, 0, world);

  if (num_temperatures > 0) {
    if (me != 0) temperatures.resize(num_temperatures);
    MPI_Bcast(temperatures.data(), num_temperatures, MPI_DOUBLE, 0, world);
  }

  // broadcast minimum energy threshold
  MPI_Bcast(&min_energy_threshold, 1, MPI_DOUBLE, 0, world);

  // broadcast complementarity map size
  int map_size = complementarity_map.size();
  MPI_Bcast(&map_size, 1, MPI_INT, 0, world);

  // broadcast complementarity map contents by packing into arrays
  if (map_size > 0) {
    std::vector<tagint> tags1(map_size);
    std::vector<tagint> tags2(map_size);
    std::vector<double> all_energies(map_size * num_temperatures);

    // rank 0 packs map data into flat arrays for MPI broadcast
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

    // non-root processors unpack arrays into their local map
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
      fprintf(screen, "Fix dnafold/bond/pre: Read %d temperatures and %d complementarity pairs from '%s'\n",
              num_temperatures, map_size, complementarity_file);
    if (logfile)
      fprintf(logfile, "Fix dnafold/bond/pre: Read %d temperatures and %d complementarity pairs from '%s'\n",
              num_temperatures, map_size, complementarity_file);
  }
}

/* ----------------------------------------------------------------------
   Get temperature-interpolated binding energy for an atom pair.
   Returns -BIG if pair is not in complementarity map (not complementary).
------------------------------------------------------------------------- */

double FixDnafoldBondPre::get_interpolated_energy(tagint tag_i, tagint tag_j)
{
  // ensure tag1 < tag2 for consistent map lookup
  tagint tag1 = (tag_i < tag_j) ? tag_i : tag_j;
  tagint tag2 = (tag_i < tag_j) ? tag_j : tag_i;

  // lookup pair in complementarity map
  auto it = complementarity_map.find(std::make_pair(tag1, tag2));
  if (it == complementarity_map.end()) return -BIG;

  const std::vector<double> &energies = it->second;

  // get current temperature from the LAMMPS variable
  double T = input->variable->compute_equal(tvar_index);

  // verify temperature is within the valid range from file
  if (T < temperatures.front() || T > temperatures.back()) {
    error->all(FLERR, fmt::format(
      "Fix dnafold/bond/pre: Temperature {} outside complementarity file range [{}, {}]",
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

  return energies.back();
}

/* ----------------------------------------------------------------------
   Check if two atoms are complementary at current temperature.
   Returns true if their interpolated energy meets the minimum threshold.
------------------------------------------------------------------------- */

bool FixDnafoldBondPre::is_complementary(tagint tag_i, tagint tag_j)
{
  double energy = get_interpolated_energy(tag_i, tag_j);
  return energy >= min_energy_threshold;
}

/* ----------------------------------------------------------------------
   Check if atom i has a dummy bond to atom j.
------------------------------------------------------------------------- */

bool FixDnafoldBondPre::has_dummy_bond(int i, int j)
{
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  tagint jtag = atom->tag[j];

  for (int k = 0; k < num_bond[i]; k++) {
    if (bond_atom[i][k] == jtag && bond_type[i][k] == dummy_bond_type) return true;
  }
  return false;
}

/* ----------------------------------------------------------------------
   Check if atom i has any bond (of any type) to atom j.
------------------------------------------------------------------------- */

bool FixDnafoldBondPre::has_any_bond(int i, int j)
{
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  tagint jtag = atom->tag[j];

  for (int k = 0; k < num_bond[i]; k++) {
    if (bond_atom[i][k] == jtag) return true;
  }
  return false;
}

/* ----------------------------------------------------------------------
   Main function called every nevery timesteps.
   Two passes: (1) remove invalid dummy bonds, (2) create new dummy bonds.
------------------------------------------------------------------------- */

void FixDnafoldBondPre::post_integrate()
{
  int i, j, ii, jj, inum, jnum, k;
  int *ilist, *jlist, *numneigh, **firstneigh;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int itype, jtype;
  tagint itag, jtag;

  if (update->ntimestep % nevery) return;

  // acquire updated ghost atom positions and rebuild occasional neighbor list
  comm->forward_comm();
  neighbor->build_one(list);

  // get neighbor list data
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // get atom data pointers
  int nlocal = atom->nlocal;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;

  // reset per-step counters
  create_count = 0;
  remove_count = 0;

  // ========== FIRST PASS: Remove dummy bonds that should no longer exist ==========
  // A dummy bond is removed if:
  // - The bonded atom is no longer accessible (not in ghost list)
  // - The distance exceeds the cutoff
  // - The pair is no longer complementary at the current temperature

  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    // only process scaffold (type 1) and staple (type 2) atoms
    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    itag = tag[i];

    // loop through bonds, removing ones that fail criteria
    // use while loop since we may remove bonds and shift array
    k = 0;
    while (k < num_bond[i]) {
      // skip non-dummy bonds (e.g., backbone bonds, hyb bonds)
      if (bond_type[i][k] != dummy_bond_type) {
        k++;
        continue;
      }

      // find partner atom by tag
      jtag = bond_atom[i][k];
      j = atom->map(jtag);

      // partner not found in local+ghost atoms - remove bond
      if (j < 0) {
        for (int m = k; m < num_bond[i] - 1; m++) {
          bond_type[i][m] = bond_type[i][m+1];
          bond_atom[i][m] = bond_atom[i][m+1];
        }
        num_bond[i]--;
        remove_count++;
        continue;
      }

      // skip if partner is not the expected opposite type
      jtype = type[j];
      if (!((itype == iatomtype && jtype == jatomtype) ||
            (itype == jatomtype && jtype == iatomtype))) {
        k++;
        continue;
      }

      // calculate distance with minimum image convention for periodic boundaries
      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
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

      // remove bond if beyond cutoff or no longer complementary at current T
      bool should_remove = (rsq > cutoff_sq) || !is_complementary(itag, jtag);

      if (should_remove) {
        for (int m = k; m < num_bond[i] - 1; m++) {
          bond_type[i][m] = bond_type[i][m+1];
          bond_atom[i][m] = bond_atom[i][m+1];
        }
        num_bond[i]--;
        remove_count++;
        continue;
      }

      k++;
    }
  }

  // ========== SECOND PASS: Create new dummy bonds for eligible pairs ==========
  // A new dummy bond is created if:
  // - The pair is within cutoff distance
  // - The pair is complementary at current temperature
  // - No bond of any type exists between them
  // - Both atoms have sufficient hybridization capacity

  int *hyb_status = atom->ivector[hyb_status_index];
  int *size = atom->ivector[size_index];

  // loop over neighbor list to find candidate pairs
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    if (!(mask[i] & groupbit)) continue;

    // only process scaffold and staple atoms
    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    // check each neighbor of atom i
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;

      // must be opposite types (scaffold-staple pair)
      jtype = type[j];
      if (!((itype == iatomtype && jtype == jatomtype) ||
            (itype == jatomtype && jtype == iatomtype))) continue;

      if (i == j) continue;

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

      // check if pair is complementary at current temperature
      jtag = tag[j];
      if (!is_complementary(itag, jtag)) continue;

      // skip if already bonded (any bond type)
      if (has_any_bond(i, j)) continue;

      // check hybridization capacity constraint
      // min_size is the smaller of the two atoms' sizes (1 for half-bead, 2 for whole)
      // each atom can have at most MAX_HYB_CAPACITY hybridization slots used
      int min_size = (size[i] < size[j]) ? size[i] : size[j];
      if (hyb_status[i] + min_size > MAX_HYB_CAPACITY) continue;
      if (hyb_status[j] + min_size > MAX_HYB_CAPACITY) continue;

      // with newton_bond on, only store bond on lower-tagged atom to avoid duplicates
      if (itag > jtag) continue;

      // create the dummy bond
      if (num_bond[i] >= atom->bond_per_atom)
        error->one(FLERR, "Too many bonds per atom in fix dnafold/bond/pre");

      bond_type[i][num_bond[i]] = dummy_bond_type;
      bond_atom[i][num_bond[i]] = jtag;
      num_bond[i]++;
      create_count++;
    }
  }

  // ========== Finalize: aggregate counts and rebuild special lists ==========

  // accumulate counts across all MPI processors
  int create_count_all, remove_count_all;
  MPI_Allreduce(&create_count, &create_count_all, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&remove_count, &remove_count_all, 1, MPI_INT, MPI_SUM, world);

  create_count_total += create_count_all;
  remove_count_total += remove_count_all;
  create_count = create_count_all;
  remove_count = remove_count_all;

  // if any bonds changed, rebuild special neighbor lists and trigger reneighboring
  if (create_count > 0 || remove_count > 0) {
    // suppress verbose output from Special::build()
    FILE *screen_save = screen;
    FILE *logfile_save = logfile;
    screen = nullptr;
    logfile = nullptr;

    Special special(lmp);
    special.build();

    screen = screen_save;
    logfile = logfile_save;

    next_reneighbor = update->ntimestep;
  }
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondPre::compute_vector(int n)
{
  // output vector: [0]=created, [1]=removed, [2]=total_created, [3]=total_removed
  if (n == 0) return (double) create_count;
  if (n == 1) return (double) remove_count;
  if (n == 2) return (double) create_count_total;
  return (double) remove_count_total;
}

/* ---------------------------------------------------------------------- */

double FixDnafoldBondPre::memory_usage()
{
  // estimate memory used by complementarity map and temperature array
  double bytes = complementarity_map.size() * (2 * sizeof(tagint) + num_temperatures * sizeof(double) + 32);
  bytes += temperatures.size() * sizeof(double);
  return bytes;
}
