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

   fix dnafold/angle/lp dynamically updates the bending stiffness (K) for
   angle type 1 based on the current temperature and persistence length
   data from a file.

   Formula: k_theta = Lp * kB * T / r12, then K_angle = k_theta / 2
------------------------------------------------------------------------- */

#include "fix_dnafold_angle_lp.h"

#include "angle.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixDnafoldAngleLp::FixDnafoldAngleLp(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), lp_file(nullptr), tvar(nullptr), k_angle(nullptr)
{
  // syntax: fix ID group dnafold/angle/lp nevery r12 lp_file v_temperature
  if (narg != 7) error->all(FLERR, "Illegal fix dnafold/angle/lp command");

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

  // parse nevery - how often to update angle parameters
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "Illegal fix dnafold/angle/lp nevery value");

  // parse r12 - characteristic length for conversion
  r12 = utils::numeric(FLERR, arg[4], false, lmp);
  if (r12 <= 0.0) error->all(FLERR, "Illegal fix dnafold/angle/lp r12 value");

  // parse persistence length file path
  lp_file = utils::strdup(arg[5]);

  // parse temperature variable name (must start with v_)
  if (strncmp(arg[6], "v_", 2) != 0)
    error->all(FLERR, "Temperature variable for fix dnafold/angle/lp must start with v_");
  tvar = utils::strdup(arg[6] + 2);
  tvar_index = -1;

  // initialize data
  num_data_points = 0;

  // read persistence length data from file
  read_lp_file();
}

/* ---------------------------------------------------------------------- */

FixDnafoldAngleLp::~FixDnafoldAngleLp()
{
  delete[] lp_file;
  delete[] tvar;
}

/* ---------------------------------------------------------------------- */

int FixDnafoldAngleLp::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDnafoldAngleLp::init()
{
  // find and validate the temperature variable
  tvar_index = input->variable->find(tvar);
  if (tvar_index < 0)
    error->all(FLERR, "Variable {} for fix dnafold/angle/lp does not exist", tvar);
  if (!input->variable->equalstyle(tvar_index))
    error->all(FLERR, "Variable {} for fix dnafold/angle/lp must be equal-style", tvar);

  // validate angle style is defined
  if (force->angle == nullptr)
    error->all(FLERR, "Must define angle_style before fix dnafold/angle/lp");

  // extract pointer to angle K parameter array
  int dim;
  void *ptr = force->angle->extract("k", dim);
  if (ptr == nullptr)
    error->all(FLERR, "Angle style does not support 'k' parameter extraction for fix dnafold/angle/lp");

  k_angle = (double *) ptr;

  // validate we have at least 1 angle type
  if (atom->nangletypes < 1)
    error->all(FLERR, "Fix dnafold/angle/lp requires at least 1 angle type");
}

/* ---------------------------------------------------------------------- */

void FixDnafoldAngleLp::setup(int /* vflag */)
{
  // perform initial update of angle parameters (bypasses nevery check)
  update_angle_k();
}

/* ----------------------------------------------------------------------
   Read persistence length data file.
   Format: two columns - temperature and persistence length
   Lines starting with # are comments
------------------------------------------------------------------------- */

void FixDnafoldAngleLp::read_lp_file()
{
  // rank 0 reads the file, then broadcasts data to all processors
  if (me == 0) {
    std::ifstream file(lp_file);
    if (!file.is_open())
      error->one(FLERR, fmt::format("Cannot open persistence length file '{}'", lp_file));

    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
      line_num++;

      // skip empty lines
      if (line.empty()) continue;

      // skip comment lines starting with #
      size_t start = line.find_first_not_of(" \t");
      if (start == std::string::npos) continue;
      if (line[start] == '#') continue;

      // parse temperature and persistence length
      std::istringstream iss(line);
      double temp, lp;

      if (!(iss >> temp >> lp)) {
        error->one(FLERR, fmt::format("Invalid format in '{}' at line {}", lp_file, line_num));
      }

      // validate values
      if (temp < 0.0) {
        error->one(FLERR, fmt::format("Negative temperature in '{}' at line {}", lp_file, line_num));
      }
      if (lp < 0.0) {
        error->one(FLERR, fmt::format("Negative persistence length in '{}' at line {}", lp_file, line_num));
      }

      temperatures.push_back(temp + 273.15);  // convert Celsius to Kelvin
      persistence_lengths.push_back(lp);
    }

    file.close();

    if (temperatures.empty()) {
      error->one(FLERR, fmt::format("No data found in persistence length file '{}'", lp_file));
    }

    // sort data by temperature (ascending order) to enable interpolation
    // create index array and sort by temperature
    std::vector<size_t> indices(temperatures.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

    std::sort(indices.begin(), indices.end(),
              [this](size_t a, size_t b) { return temperatures[a] < temperatures[b]; });

    // reorder both arrays according to sorted indices
    std::vector<double> sorted_temps(temperatures.size());
    std::vector<double> sorted_lps(persistence_lengths.size());
    for (size_t i = 0; i < indices.size(); i++) {
      sorted_temps[i] = temperatures[indices[i]];
      sorted_lps[i] = persistence_lengths[indices[i]];
    }
    temperatures = std::move(sorted_temps);
    persistence_lengths = std::move(sorted_lps);

    num_data_points = temperatures.size();
  }

  // broadcast number of data points
  MPI_Bcast(&num_data_points, 1, MPI_INT, 0, world);

  if (num_data_points < 1) {
    error->all(FLERR, "Persistence length file contains no valid data");
  }

  // resize vectors on non-root processors
  if (me != 0) {
    temperatures.resize(num_data_points);
    persistence_lengths.resize(num_data_points);
  }

  // broadcast data arrays
  MPI_Bcast(temperatures.data(), num_data_points, MPI_DOUBLE, 0, world);
  MPI_Bcast(persistence_lengths.data(), num_data_points, MPI_DOUBLE, 0, world);

  // log info about loaded data
  if (me == 0) {
    if (screen)
      fprintf(screen, "Fix dnafold/angle/lp: Read %d temperature-Lp pairs from '%s'\n",
              num_data_points, lp_file);
    if (logfile)
      fprintf(logfile, "Fix dnafold/angle/lp: Read %d temperature-Lp pairs from '%s'\n",
              num_data_points, lp_file);
  }
}

/* ----------------------------------------------------------------------
   Get interpolated persistence length for a given temperature.
   Uses linear interpolation between data points.
   If T < min temp, extrapolates from the lowest two data points.
   Returns -1.0 if T > max temp (signals to set K to zero).
------------------------------------------------------------------------- */

double FixDnafoldAngleLp::get_persistence_length(double T)
{
  // if temperature exceeds the table maximum, return -1 to signal K=0
  if (T > temperatures.back()) {
    return -1.0;
  }

  // if temperature is below the table minimum, extrapolate from lowest two points
  if (T < temperatures.front()) {
    if (num_data_points < 2) {
      // only one data point - just use it
      return persistence_lengths.front();
    }
    double t0 = temperatures[0];
    double t1 = temperatures[1];
    double lp0 = persistence_lengths[0];
    double lp1 = persistence_lengths[1];
    double frac = (T - t0) / (t1 - t0);  // will be negative
    return lp0 + frac * (lp1 - lp0);
  }

  // find bracketing temperatures and interpolate
  for (int i = 0; i < num_data_points - 1; i++) {
    if (T >= temperatures[i] && T <= temperatures[i + 1]) {
      double t0 = temperatures[i];
      double t1 = temperatures[i + 1];
      double lp0 = persistence_lengths[i];
      double lp1 = persistence_lengths[i + 1];
      double frac = (T - t0) / (t1 - t0);
      return lp0 + frac * (lp1 - lp0);
    }
  }

  // exact match at last temperature
  return persistence_lengths.back();
}

/* ----------------------------------------------------------------------
   Called every nevery timesteps to update angle K.
------------------------------------------------------------------------- */

void FixDnafoldAngleLp::post_integrate()
{
  if (update->ntimestep % nevery) return;
  update_angle_k();
}

/* ----------------------------------------------------------------------
   Core function to update angle K based on current temperature.
   Called from setup() and post_integrate().
------------------------------------------------------------------------- */

void FixDnafoldAngleLp::update_angle_k()
{
  // get current temperature from the LAMMPS variable
  double T = input->variable->compute_equal(tvar_index);

  // get interpolated persistence length at current temperature
  // returns -1.0 if T exceeds the table maximum
  double Lp = get_persistence_length(T);

  if (Lp < 0.0) {
    // temperature exceeds table maximum - set angle stiffness to zero
    k_angle[1] = 0.0;
  } else {
    // calculate k_theta = Lp * kB * T / r12
    double k_theta = Lp * BOLTZMANN * T / r12;

    // set angle K for type 1 to k_theta / 2
    k_angle[1] = k_theta / 2.0;
  }
}

/* ---------------------------------------------------------------------- */

double FixDnafoldAngleLp::memory_usage()
{
  double bytes = temperatures.size() * sizeof(double);
  bytes += persistence_lengths.size() * sizeof(double);
  return bytes;
}
