#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <eigen3/Eigen/Dense>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "pcl_helper_functions.h"

using namespace std;
using namespace Eigen;

const int STATE_SIZE = 7;
const int NPOP_PARTICLES = 200;

typedef Matrix<float, STATE_SIZE, NPOP_PARTICLES> StateMatrix;
typedef Matrix<float, STATE_SIZE, 1> MeanVector;
typedef Matrix<float, NPOP_PARTICLES, 1> WeightVector;
typedef Matrix<float, 1, NPOP_PARTICLES> RowVector;

class particle_filter
{
private:
    gsl_rng* rng;
    StateMatrix stateMatrix;
    WeightVector weightVector;

    RowVector generate_uniform(float a, float b);
    RowVector generate_gaussian(float std);
public:
    particle_filter();
    void initialize(float x, float y, float z, float xl, float yl, float zl, float o);
    void resample_particles();
    void print_state();
    void update_state();
    void update_weights(pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud);
    StateMatrix get_state();
    MeanVector compute_mean();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr generate_box(float x, float y, float z, float xl, float yl, float zl, float o, float rate, float noise);

};

#endif // PARTICLE_FILTER_H
