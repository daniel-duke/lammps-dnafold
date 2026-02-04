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

#include "fix_dnafold_bond_hyb.h"

#include "atom.h"
#include "special.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
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

FixDnafoldBondHyb::FixDnafoldBondHyb(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), complementarity_file(nullptr), tvar(nullptr),
  partner(nullptr), finalpartner(nullptr), partnerbtype(nullptr), distsq(nullptr), partner_energy(nullptr)
{
  if (narg != 8) error->all(FLERR,"Illegal fix dnafold/bond/hyb command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix dnafold/bond/hyb command");

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  vector_flag = 1;
  size_vector = 4;  // create, total create, downgrade, total downgrade
  global_freq = 1;
  extvector = 0;
  
  comm_forward = 4;  // partner tag, bond type, distance squared, and energy_depth
  comm_reverse = 4;  // partner tag, bond type, distance squared, and energy_depth

  // Bonds form between type 1 and type 2 atoms only
  iatomtype = 1;
  jatomtype = 2;

  cutoffsq = utils::numeric(FLERR,arg[4],false,lmp);
  if (cutoffsq <= 0.0) error->all(FLERR,"Illegal cutoff");
  cutoffsq = cutoffsq * cutoffsq;

  dummy_btype = utils::inumeric(FLERR,arg[5],false,lmp);
  if (dummy_btype <= 0) error->all(FLERR,"Illegal dummy bond type");

  complementarity_file = utils::strdup(arg[6]);

  // Parse temperature variable (must start with v_)
  if (strncmp(arg[7], "v_", 2) != 0)
    error->all(FLERR,"Temperature variable for fix dnafold/bond/hyb must start with v_");
  tvar = utils::strdup(arg[7] + 2);  // skip "v_" prefix
  tvar_index = -1;  // will be set in init()

  createcount = 0;
  downgradecount = 0;
  createcounttotal = 0;
  downgradecounttotal = 0;

  nmax = 0;
  num_temperatures = 0;

  // Read complementarity file - only rank 0 reads, then broadcasts
  read_complementarity_file();
}

FixDnafoldBondHyb::~FixDnafoldBondHyb()
{
  delete[] complementarity_file;
  delete[] tvar;
  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(partnerbtype);
  memory->destroy(distsq);
  memory->destroy(partner_energy);
}

int FixDnafoldBondHyb::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

void FixDnafoldBondHyb::init()
{
  // Find the i_hyb_status property (now an integer)
  int flag_hyb, cols_hyb;
  property_flag_index = atom->find_custom("hyb_status", flag_hyb, cols_hyb);
  if (property_flag_index < 0)
    error->all(FLERR,fmt::format("Could not find property 'i_hyb_status'"));
  if (flag_hyb != 0)
    error->all(FLERR,fmt::format("Property 'i_hyb_status' must be integer"));

  // Find the i_size property (now an integer)
  int flag_size, cols_size;
  size_index = atom->find_custom("size", flag_size, cols_size);
  if (size_index < 0)
    error->all(FLERR,fmt::format("Could not find property 'i_size'"));
  if (flag_size != 0)
    error->all(FLERR,fmt::format("Property 'i_size' must be integer"));

  // Find and validate temperature variable
  tvar_index = input->variable->find(tvar);
  if (tvar_index < 0)
    error->all(FLERR,"Variable {} for fix dnafold/bond/hyb does not exist", tvar);
  if (!input->variable->equalstyle(tvar_index))
    error->all(FLERR,"Variable {} for fix dnafold/bond/hyb must be equal-style", tvar);

  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix dnafold/bond/hyb with non-molecular system");

  if (force->bond == nullptr)
    error->all(FLERR,"Must define bond_style");

  if (force->newton_bond == 0)
    error->all(FLERR,"Fix dnafold/bond/hyb requires newton bond on");
}

void FixDnafoldBondHyb::setup(int /* vflag */)
{
  post_integrate();
}

void FixDnafoldBondHyb::read_complementarity_file()
{
  // Only rank 0 reads the file
  if (me == 0) {
    std::ifstream file(complementarity_file);
    if (!file.is_open())
      error->one(FLERR,fmt::format("Cannot open complementarity file '{}'", complementarity_file));

    std::string line;
    int line_num = 0;
    enum Section { NONE, TEMPERATURES, TYPES, PAIRS };
    Section current_section = NONE;

    while (std::getline(file, line)) {
      line_num++;

      // Skip empty lines
      if (line.empty()) continue;

      // Skip comment lines (lines starting with #)
      if (line[0] == '#') continue;

      // Trim leading whitespace for section detection
      size_t start = line.find_first_not_of(" \t");
      if (start == std::string::npos) continue;  // All whitespace
      std::string trimmed = line.substr(start);

      // Check for section headers
      if (trimmed == "TEMPERATURES") {
        current_section = TEMPERATURES;
        continue;
      } else if (trimmed == "TYPES") {
        current_section = TYPES;
        continue;
      } else if (trimmed == "PAIRS") {
        current_section = PAIRS;
        if (temperatures.empty()) {
          error->one(FLERR,fmt::format("TEMPERATURES section must appear before PAIRS in '{}'",
                                       complementarity_file));
        }
        continue;
      }

      // Parse TEMPERATURES section
      if (current_section == TEMPERATURES) {
        std::istringstream iss(line);
        double temp;
        while (iss >> temp) {
          temperatures.push_back(temp);
        }
        if (temperatures.empty()) {
          error->one(FLERR,fmt::format("Invalid TEMPERATURES format in '{}' at line {}",
                                       complementarity_file, line_num));
        }
        num_temperatures = temperatures.size();
        continue;
      }

      // Parse TYPES section
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

        // Store energy level and bond type
        energy_levels.push_back(std::make_pair(energy, btype));
        continue;
      }

      // Parse PAIRS section
      if (current_section == PAIRS) {
        std::istringstream iss(line);
        tagint tag1, tag2;

        if (!(iss >> tag1 >> tag2)) {
          error->one(FLERR,fmt::format("Invalid PAIRS format in '{}' at line {}",
                                       complementarity_file, line_num));
        }

        // Read energy values for each temperature
        std::vector<double> energies;
        double energy;
        while (iss >> energy) {
          energies.push_back(energy);
        }

        if ((int)energies.size() != num_temperatures) {
          error->one(FLERR,fmt::format("PAIRS line {} in '{}' has {} energies, expected {}",
                                       line_num, complementarity_file,
                                       energies.size(), num_temperatures));
        }

        // Ensure tag1 < tag2 for consistent lookup
        if (tag1 > tag2) std::swap(tag1, tag2);

        // Store in map
        complementarity_map[std::make_pair(tag1, tag2)] = energies;

      } else if (current_section == NONE) {
        error->one(FLERR,fmt::format("Line {} in '{}' appears before any section header",
                                     line_num, complementarity_file));
      }
    }

    file.close();

    if (temperatures.empty()) {
      error->one(FLERR,fmt::format("No TEMPERATURES section found in '{}'", complementarity_file));
    }

    if (energy_levels.empty()) {
      error->one(FLERR,fmt::format("No bond types defined in TYPES section of '{}'",
                                   complementarity_file));
    }

    if (complementarity_map.empty()) {
      error->warning(FLERR,"Complementarity file '{}' contains no valid pairs",
                     complementarity_file);
    }

    // Sort energy levels by energy (descending) for efficient lookup
    // Higher energy = stronger bond
    std::sort(energy_levels.begin(), energy_levels.end(),
              [](const std::pair<double,int> &a, const std::pair<double,int> &b) {
                return a.first > b.first;  // Descending order
              });
  }

  // Broadcast num_temperatures and temperatures
  MPI_Bcast(&num_temperatures, 1, MPI_INT, 0, world);

  if (num_temperatures > 0) {
    if (me != 0) temperatures.resize(num_temperatures);
    MPI_Bcast(temperatures.data(), num_temperatures, MPI_DOUBLE, 0, world);
  }

  // Broadcast energy levels
  int num_levels = energy_levels.size();
  MPI_Bcast(&num_levels, 1, MPI_INT, 0, world);

  if (num_levels > 0) {
    if (me != 0) energy_levels.resize(num_levels);

    // Prepare arrays for broadcast
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

    if (me != 0) {
      for (int i = 0; i < num_levels; i++) {
        energy_levels[i] = std::make_pair(level_energies[i], btypes[i]);
      }
    }
  }

  // Broadcast complementarity map size
  int map_size = complementarity_map.size();
  MPI_Bcast(&map_size, 1, MPI_INT, 0, world);

  // Broadcast map contents
  if (map_size > 0) {
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

  if (me == 0) {
    if (screen)
      fprintf(screen,"Fix dnafold/bond/hyb: Read %d temperatures, %d bond types, and %d complementarity pairs from '%s'\n",
              num_temperatures, num_levels, map_size, complementarity_file);
    if (logfile)
      fprintf(logfile,"Fix dnafold/bond/hyb: Read %d temperatures, %d bond types, and %d complementarity pairs from '%s'\n",
              num_temperatures, num_levels, map_size, complementarity_file);
  }
}

double FixDnafoldBondHyb::get_interpolated_energy(tagint tag_i, tagint tag_j)
{
  // Ensure tag1 < tag2 for consistent lookup
  tagint tag1 = (tag_i < tag_j) ? tag_i : tag_j;
  tagint tag2 = (tag_i < tag_j) ? tag_j : tag_i;

  auto it = complementarity_map.find(std::make_pair(tag1, tag2));
  if (it == complementarity_map.end()) return -BIG;  // Not complementary

  const std::vector<double>& energies = it->second;

  // Get current temperature from variable
  double T = input->variable->compute_equal(tvar_index);

  // Check bounds - error if outside range
  if (T < temperatures.front() || T > temperatures.back()) {
    error->all(FLERR, fmt::format(
      "Fix dnafold/bond/hyb: Temperature {} outside complementarity file range [{}, {}]",
      T, temperatures.front(), temperatures.back()));
  }

  // Find bracketing temperatures and interpolate
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

  return energies.back();  // Should not reach here, but return last value
}

int FixDnafoldBondHyb::get_bond_type(tagint tag_i, tagint tag_j)
{
  double target_energy = get_interpolated_energy(tag_i, tag_j);

  // Not complementary
  if (target_energy < -BIG/2) return 0;

  // If target_energy < lowest available energy, return 0 (no bond)
  // energy_levels is sorted descending, so .back() is the lowest (weakest)
  if (target_energy < energy_levels.back().first) {
    return 0;  // Energy too low, don't create bond
  }

  // Find closest energy level
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

double FixDnafoldBondHyb::get_energy_depth(tagint tag_i, tagint tag_j)
{
  double energy = get_interpolated_energy(tag_i, tag_j);

  // Not complementary - return very large value (for "worst" comparison)
  if (energy < -BIG/2) return BIG;

  return energy;
}

void FixDnafoldBondHyb::post_integrate()
{
  int i,j,m;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int itype, jtype, btype_ij;
  tagint itag, jtag;

  if (update->ntimestep % nevery) return;

  int *hyb_status = atom->ivector[property_flag_index];
  int *size = atom->ivector[size_index];
  
  // Acquire updated ghost atom positions and properties
  // This ensures ghosts have current coordinates and custom properties
  comm->forward_comm();

  // Resize partner arrays if needed
  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(finalpartner);
    memory->destroy(partnerbtype);
    memory->destroy(distsq);
    memory->destroy(partner_energy);
    nmax = atom->nmax;
    memory->create(partner, nmax, "dnafold/bond/hyb:partner");
    memory->create(finalpartner, nmax, "dnafold/bond/hyb:finalpartner");
    memory->create(partnerbtype, nmax, "dnafold/bond/hyb:partnerbtype");
    memory->create(distsq, nmax, "dnafold/bond/hyb:distsq");
    memory->create(partner_energy, nmax, "dnafold/bond/hyb:partner_energy");
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;

  // Initialize partner arrays for all atoms (local + ghost)
  for (i = 0; i < nall; i++) {
    partner[i] = 0;
    finalpartner[i] = 0;
    partnerbtype[i] = 0;
    distsq[i] = BIG;
    partner_energy[i] = BIG;
  }

  // First pass: check existing hyb bonds for distance and energy-based updates
  // - If too far: downgrade to dummy
  // - If energy dropped below threshold: downgrade to dummy
  // - If energy changed to different bond type: update bond type
  int downgradecount = 0;
  int updatecount = 0;
  std::vector<tagint> downgraded_type1_tags;  // Track type 1 atoms that were downgraded

  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    // Only check atoms with hyb_status > 0 (have hybridized bonds)
    if (hyb_status[i] == 0) continue;

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // Loop through i's bonds looking for hyb bonds (non-dummy bonds)
    int ib = 0;
    while (ib < num_bond[i]) {
      // Skip dummy bonds - they're handled in the second pass
      if (bond_type[i][ib] == dummy_btype) {
        ib++;
        continue;
      }

      // This is a hyb bond - get partner info
      jtag = bond_atom[i][ib];
      j = atom->map(jtag);
      if (j < 0) {
        // Partner not found - this indicates ghost cutoff is too small
        error->one(FLERR,"Fix dnafold/bond/hyb: Bonded atom not found in ghost atoms. "
                         "Increase communication cutoff with 'comm_modify cutoff'");
      }

      // Check distance with minimum image convention
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

      // Get the correct bond type at current temperature
      int correct_btype = get_bond_type(itag, jtag);

      // Determine action: downgrade to dummy, update bond type, or keep as-is
      bool should_downgrade = false;

      if (rsq > cutoffsq) {
        // Too far - downgrade to dummy
        should_downgrade = true;
      } else if (correct_btype == 0) {
        // Energy dropped below threshold at current T - downgrade to dummy
        should_downgrade = true;
      }

      if (should_downgrade) {
        // Downgrade to dummy bond
        bond_type[i][ib] = dummy_btype;

        // Calculate min_size and subtract from hyb_status for both atoms
        int min_size = (size[i] < size[j]) ? size[i] : size[j];
        hyb_status[i] -= min_size;

        // Also update j's hyb_status if j is local
        if (j < nlocal) hyb_status[j] -= min_size;

        // If i is type 1, mark for angle removal
        if (itype == iatomtype) {
          downgraded_type1_tags.push_back(itag);
        }

        downgradecount++;
      } else if (bond_type[i][ib] != correct_btype) {
        // Bond type changed due to temperature - update it
        // This could be upgrade (stronger) or downgrade (weaker) within hyb bonds
        bond_type[i][ib] = correct_btype;
        updatecount++;
        // Note: hyb_status doesn't change since it's still a hyb bond
      }

      ib++;
    }
  }

  // Gather all downgraded type 1 tags across processors for angle removal
  int nlocal_downgraded = downgraded_type1_tags.size();
  int ntotal_downgraded = 0;
  MPI_Allreduce(&nlocal_downgraded, &ntotal_downgraded, 1, MPI_INT, MPI_SUM, world);
  
  std::vector<tagint> all_downgraded_type1_tags;
  if (ntotal_downgraded > 0) {
    // Gather counts from all processors
    std::vector<int> recvcounts(nprocs);
    MPI_Allgather(&nlocal_downgraded, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);
    
    // Calculate displacements
    std::vector<int> displs(nprocs);
    displs[0] = 0;
    for (int p = 1; p < nprocs; p++) {
      displs[p] = displs[p-1] + recvcounts[p-1];
    }
    
    // Gather all tags
    all_downgraded_type1_tags.resize(ntotal_downgraded);
    MPI_Allgatherv(downgraded_type1_tags.data(), nlocal_downgraded, MPI_LMP_TAGINT,
                   all_downgraded_type1_tags.data(), recvcounts.data(), displs.data(),
                   MPI_LMP_TAGINT, world);
  }

  // Remove angles involving any downgraded type 1 atoms
  if (ntotal_downgraded > 0) {
    int **angle_type = atom->angle_type;
    tagint **angle_atom1 = atom->angle_atom1;
    tagint **angle_atom2 = atom->angle_atom2;
    tagint **angle_atom3 = atom->angle_atom3;
    int *num_angle = atom->num_angle;
    
    // Loop through all local atoms' angles
    for (i = 0; i < nlocal; i++) {
      int ia = 0;
      while (ia < num_angle[i]) {
        bool should_remove = false;
        
        // Check if any of the three atoms in this angle match a downgraded type 1 atom
        for (tagint downgraded_tag : all_downgraded_type1_tags) {
          if (angle_atom1[i][ia] == downgraded_tag ||
              angle_atom2[i][ia] == downgraded_tag ||
              angle_atom3[i][ia] == downgraded_tag) {
            should_remove = true;
            break;
          }
        }
        
        if (should_remove) {
          // Remove angle by shifting remaining angles down
          for (int k = ia; k < num_angle[i] - 1; k++) {
            angle_type[i][k] = angle_type[i][k+1];
            angle_atom1[i][k] = angle_atom1[i][k+1];
            angle_atom2[i][k] = angle_atom2[i][k+1];
            angle_atom3[i][k] = angle_atom3[i][k+1];
          }
          num_angle[i]--;
          // Don't increment ia since we shifted
        } else {
          ia++;
        }
      }
    }
  }

  // Second pass: upgrade dummy bonds to hyb bonds when within cutoff
  // Partner selection based on distance (closest complementary partner with dummy bond)
  for (i = 0; i < nlocal; i++) {
    
    if (!(mask[i] & groupbit)) continue;
    
    // Skip if atom i already at full capacity (hyb_status >= 2)
    if (hyb_status[i] >= 2) continue;
    
    // Atom i must be type 1 or type 2
    itype = type[i];
    if (itype != iatomtype && itype != jatomtype) continue;

    itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // Loop through i's bonds looking for dummy bonds
    for (int k = 0; k < num_bond[i]; k++) {
      // Skip if not a dummy bond
      if (bond_type[i][k] != dummy_btype) continue;
      
      // Get partner atom
      jtag = bond_atom[i][k];
      j = atom->map(jtag);
      
      if (j < 0) continue;  // Partner not found
      
      if (!(mask[j] & groupbit)) continue;
      
      // Skip if atom j already at full capacity (hyb_status >= 2)
      if (hyb_status[j] >= 2) continue;

      // Atom j must be the opposite type from i
      jtype = type[j];
      if (itype == iatomtype && jtype != jatomtype) continue;
      if (itype == jatomtype && jtype != iatomtype) continue;

      // Check distance with minimum image convention
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
      if (rsq > cutoffsq) continue;

      // Check if this pair is complementary and get bond type and energy
      btype_ij = get_bond_type(itag, jtag);
      if (btype_ij == 0) continue;  // Not complementary (shouldn't happen if dummy bond exists)
      
      double energy_ij = get_energy_depth(itag, jtag);

      // Check capacity constraint
      int min_size = (size[i] < size[j]) ? size[i] : size[j];
      if (hyb_status[i] + min_size > 2) continue;
      if (hyb_status[j] + min_size > 2) continue;

      // Update partner for atom i if this is better
      // Better = lower energy_depth (stronger bond), or same energy but closer distance
      bool better_for_i = (energy_ij < partner_energy[i]) || 
                          (energy_ij == partner_energy[i] && rsq < distsq[i]);
      if (better_for_i) {
        partner[i] = jtag;
        partnerbtype[i] = btype_ij;
        distsq[i] = rsq;
        partner_energy[i] = energy_ij;
      }
      
      // Update partner for atom j if this is better
      // This is safe even if j is a ghost - we'll communicate this back
      bool better_for_j = (energy_ij < partner_energy[j]) || 
                          (energy_ij == partner_energy[j] && rsq < distsq[j]);
      if (better_for_j) {
        partner[j] = itag;
        partnerbtype[j] = btype_ij;
        distsq[j] = rsq;
        partner_energy[j] = energy_ij;
      }
    }
  }

  // Reverse comm of partner, partnerbtype, and distsq
  // Send ghost atom data back to home processors
  // Home processor will keep the closest partner
  commflag = 1;
  if (force->newton_pair) comm->reverse_comm(this);

  // Forward comm of partner, partnerbtype, and distsq, so ghosts have final values
  // This ensures all atoms (local and ghost) know about finalized partners
  commflag = 1;
  comm->forward_comm(this);

  // Create bonds for atoms I own
  // Only if both atoms list each other as winning bond partner

  createcount = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    
    // Map partner tag to local or ghost index
    j = atom->map(partner[i]);
    if (j < 0) 
      error->one(FLERR,"Fix dnafold/bond/hyb: partner atom not found - ghost cutoff may be too small");
    
    // Both atoms must agree on being partners
    if (partner[j] != tag[i]) continue;

    // With newton_bond on, only store bond on lower-tagged atom
    // This prevents duplicate bond creation across processors
    if (tag[i] > tag[j]) continue;

    if (num_bond[i] >= atom->bond_per_atom)
      error->one(FLERR,"Too many bonds per atom in fix dnafold/bond/hyb");

    // Create the bond with the bond type from the complementarity map
    bond_type[i][num_bond[i]] = partnerbtype[i];
    bond_atom[i][num_bond[i]] = tag[j];
    num_bond[i]++;

    // Remove any dummy bond between i and j
    remove_dummy_bond(i, j);

    // Calculate min_size and add to hyb_status for both atoms
    int min_size = (size[i] < size[j]) ? size[i] : size[j];
    hyb_status[i] += min_size;
    
    // Also update j's hyb_status
    // This is safe even if j is ghost because:
    // 1. If j is local, we're setting it directly
    // 2. If j is ghost, j's home processor will also create this same bond
    //    (because both atoms agree on partnership) and set the hyb_status there
    hyb_status[j] += min_size;

    // Store final partners for bookkeeping
    finalpartner[i] = tag[j];
    
    createcount++;
  }

  int createcountall, downgradecountall, updatecountall;
  MPI_Allreduce(&createcount,&createcountall,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&downgradecount,&downgradecountall,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&updatecount,&updatecountall,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcountall;
  downgradecounttotal += downgradecountall;
  createcount = createcountall;
  downgradecount = downgradecountall;

  // If any bonds were created, downgraded, or updated, rebuild special lists and trigger reneighboring
  if (createcount || downgradecount || updatecountall) {
    // Suppress output from Special::build()
    FILE *screen_save = screen;
    FILE *logfile_save = logfile;
    screen = nullptr;
    logfile = nullptr;

    Special special(lmp);
    special.build();

    // Restore output
    screen = screen_save;
    logfile = logfile_save;

    next_reneighbor = update->ntimestep;
  }
}

int FixDnafoldBondHyb::pack_forward_comm(int n, int *list, double *buf, 
                                          int /* pbc_flag */, int * /* pbc */)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(partner[j]).d;
    buf[m++] = ubuf(partnerbtype[j]).d;
    buf[m++] = distsq[j];
    buf[m++] = partner_energy[j];
  }
  return m;
}

void FixDnafoldBondHyb::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    partner[i] = (tagint) ubuf(buf[m++]).i;
    partnerbtype[i] = (int) ubuf(buf[m++]).i;
    distsq[i] = buf[m++];
    partner_energy[i] = buf[m++];
  }
}

int FixDnafoldBondHyb::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = ubuf(partnerbtype[i]).d;
    buf[m++] = distsq[i];
    buf[m++] = partner_energy[i];
  }
  return m;
}

void FixDnafoldBondHyb::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    // Keep the better partner: lower energy_depth (stronger bond), or same energy but closer distance
    double incoming_energy = buf[m+3];
    double incoming_distsq = buf[m+2];
    
    bool better = (incoming_energy < partner_energy[j]) || 
                  (incoming_energy == partner_energy[j] && incoming_distsq < distsq[j]);
    
    if (better) {
      partner[j] = (tagint) ubuf(buf[m++]).i;
      partnerbtype[j] = (int) ubuf(buf[m++]).i;
      distsq[j] = buf[m++];
      partner_energy[j] = buf[m++];
    } else m += 4;
  }
}

void FixDnafoldBondHyb::remove_dummy_bond(int i, int j)
{
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  tagint jtag = atom->tag[j];

  // Look through i's bonds to find dummy bond to j
  for (int k = 0; k < num_bond[i]; k++) {
    if (bond_atom[i][k] == jtag && bond_type[i][k] == dummy_btype) {
      // Found dummy bond - remove it by shifting remaining bonds down
      for (int m = k; m < num_bond[i] - 1; m++) {
        bond_type[i][m] = bond_type[i][m+1];
        bond_atom[i][m] = bond_atom[i][m+1];
      }
      num_bond[i]--;
      return;
    }
  }
}

double FixDnafoldBondHyb::compute_vector(int n)
{
  if (n == 0) return (double) createcount;
  if (n == 1) return (double) createcounttotal;
  if (n == 2) return (double) downgradecount;
  return (double) downgradecounttotal;
}

double FixDnafoldBondHyb::memory_usage()
{
  double bytes = (double)nmax * 2 * sizeof(tagint);  // partner, finalpartner
  bytes += (double)nmax * sizeof(int);                // partnerbtype
  bytes += (double)nmax * 2 * sizeof(double);         // distsq, partner_energy
  
  // Estimate map memory (rough approximation)
  bytes += complementarity_map.size() * (2 * sizeof(tagint) + sizeof(int) + 32);
  
  return bytes;
}


