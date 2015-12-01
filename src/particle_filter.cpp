#include "particle_filter.h"

particle_filter::particle_filter()
{
    gsl_rng_env_setup();
    rng = gsl_rng_alloc( gsl_rng_mt19937 );
    gsl_rng_set( rng, time(NULL) );
}

void particle_filter::initialize(float x, float y, float z, float xl, float yl, float zl, float o)
{
    float posOff = 0.1; float sizeOff = 0.1; float oOff = 40;
    weightVector = WeightVector::Zero();
    stateMatrix = StateMatrix::Zero();
    stateMatrix.row(0) = generate_uniform(x-posOff, x+posOff);
    stateMatrix.row(1) = generate_uniform(y-posOff, y+posOff);
    stateMatrix.row(2) = generate_uniform(z-posOff, z+posOff);
    stateMatrix.row(3) = generate_uniform(xl-sizeOff, xl+sizeOff);
    stateMatrix.row(4) = generate_uniform(yl-sizeOff, yl+sizeOff);
    stateMatrix.row(5) = generate_uniform(zl-sizeOff, zl+sizeOff);
    stateMatrix.row(6) = generate_uniform(o-oOff, o+oOff);
}

void particle_filter::update_state(){
    float posNoise = 0.05; float sizeNoise = 0.05; float oNoise = 5;
    stateMatrix.row(0) = stateMatrix.row(0) + generate_gaussian(posNoise);
    stateMatrix.row(1) = stateMatrix.row(1) + generate_gaussian(posNoise);
    stateMatrix.row(2) = stateMatrix.row(2) + generate_gaussian(posNoise);
    stateMatrix.row(3) = stateMatrix.row(3) + generate_gaussian(sizeNoise);
    stateMatrix.row(4) = stateMatrix.row(4) + generate_gaussian(sizeNoise);
    stateMatrix.row(5) = stateMatrix.row(5) + generate_gaussian(sizeNoise);
    stateMatrix.row(6) = stateMatrix.row(6) + generate_gaussian(oNoise);
}

void particle_filter::update_weights(pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud){
    float Xstd_rot = 0.1;
    float Xstd_pos = 0.05;
    float Xstd_rmse = 0.02;
    float A_rot = -std::log(std::sqrt(2*M_PI)*Xstd_rot);
    float B_rot = -0.5/std::pow(Xstd_rot,2);
    float A_pos = -std::log(std::sqrt(2*M_PI)*Xstd_pos);
    float B_pos = -0.5/std::pow(Xstd_pos,2);
    float A_rmse = -std::log(std::sqrt(2*M_PI)*Xstd_rmse);
    float B_rmse = -0.5/std::pow(Xstd_rmse,2);

    unsigned int i;
    WeightVector sharedWeight = WeightVector::Zero();
    #pragma omp parallel for private(i) shared(sharedWeight)
    for(i = 0; i < NPOP_PARTICLES; i++){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr particleCloud = generate_box(stateMatrix(0,i),stateMatrix(1,i),stateMatrix(2,i),stateMatrix(3,i),stateMatrix(4,i),stateMatrix(5,i),stateMatrix(6,i),0.05,0);
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setInputCloud(particleCloud);
        icp.setInputTarget(inputCloud);
        pcl::PointCloud<pcl::PointXYZRGB> Final;
        icp.align(Final);
        float rmse = icp.getFitnessScore();
        Matrix4f transform = icp.getFinalTransformation();
        Matrix3f rot = (transform.block(0,0,3,3));
        Vector3f ea = rot.eulerAngles(0, 1, 2) * (180.0 / M_PI);

        float finalrot = ea(2);
        if(abs(ea(0))>178){
            if(finalrot < 0)
                finalrot = 180 + finalrot;
            else if(finalrot >= 0)
                finalrot = finalrot - 180;
        }
        finalrot = finalrot * (M_PI / 180);

        sharedWeight(i) =
                (A_rot + B_rot * (pow((finalrot*10),2))) +
                (A_pos + B_pos * (pow(transform(0,3)*10,2))) +
                (A_pos + B_pos * (pow(transform(1,3)*10,2))) +
                (A_pos + B_pos * (pow(transform(2,3)*10,2))) +
                (A_rmse + B_rmse * (pow((rmse*100),2)))
        ;

    }
    weightVector = sharedWeight;
    std::cout << "Done" << std::endl;
}

MeanVector particle_filter::compute_mean(){
    MeanVector meanVector = MeanVector::Zero();
    for(int i=0;i<STATE_SIZE;i++){
        meanVector(i) = stateMatrix.row(i).mean();
    }
    return meanVector;
}

void particle_filter::print_state(){
    std::cout << stateMatrix.row(0) << "\n";
    std::cout << stateMatrix.row(0).mean() << "\n";
}

StateMatrix particle_filter::get_state(){
    return stateMatrix;
}

RowVector particle_filter::generate_uniform(float a, float b){
    srand((unsigned int) time(0));
    RowVector rowVector = (b-a)*((RowVector::Random() + RowVector::Constant(1)) / 2) + RowVector::Constant(a);
    return rowVector;
}

RowVector particle_filter::generate_gaussian(float std){
    RowVector rowVector = RowVector::Zero();
    for(int i=0; i<NPOP_PARTICLES; ++i){
        rowVector(0,i) = gsl_ran_gaussian(rng, std);
    }
    return rowVector;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr particle_filter::generate_box(float x, float y, float z, float xl, float yl, float zl, float o, float rate, float noise){
    pcl::PointCloud<pcl::PointXYZ> boxCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(float xPos=(x-(xl/2)); xPos<=(x+(xl/2)); xPos+=rate){
        pcl::PointXYZ point1(xPos, y-(yl/2), z+(zl/2)); boxCloud.push_back(point1);
        pcl::PointXYZ point2(xPos, y+(yl/2), z-(zl/2)); boxCloud.push_back(point2);
        pcl::PointXYZ point3(xPos, y+(yl/2), z+(zl/2)); boxCloud.push_back(point3);
        pcl::PointXYZ point4(xPos, y-(yl/2), z-(zl/2)); boxCloud.push_back(point4);
    }
    for(float yPos=(y-(yl/2)); yPos<=(y+(yl/2)); yPos+=rate){
        pcl::PointXYZ point1(x-(xl/2), yPos, z+(zl/2)); boxCloud.push_back(point1);
        pcl::PointXYZ point2(x+(xl/2), yPos, z+(zl/2)); boxCloud.push_back(point2);
        pcl::PointXYZ point3(x-(xl/2), yPos, z-(zl/2)); boxCloud.push_back(point3);
        pcl::PointXYZ point4(x+(xl/2), yPos, z-(zl/2)); boxCloud.push_back(point4);
    }
    for(float zPos=(z-(zl/2)); zPos<=(z+(zl/2)); zPos+=rate){
        pcl::PointXYZ point1(x-(xl/2), y+(yl/2), zPos); boxCloud.push_back(point1);
        pcl::PointXYZ point2(x+(xl/2), y+(yl/2), zPos); boxCloud.push_back(point2);
        pcl::PointXYZ point3(x-(xl/2), y-(yl/2), zPos); boxCloud.push_back(point3);
        pcl::PointXYZ point4(x+(xl/2), y-(yl/2), zPos); boxCloud.push_back(point4);
    }

    pcl::transformPointCloud(boxCloud, boxCloud, pcl_helper_functions::getTransformMatrix(0,0,0,-x,-y,-z));
    pcl::transformPointCloud(boxCloud, boxCloud, pcl_helper_functions::getTransformMatrix(o,0,0,0,0,0));
    pcl::transformPointCloud(boxCloud, boxCloud, pcl_helper_functions::getTransformMatrix(0,0,0,x,y,z));
    pcl::copyPointCloud(boxCloud,*outputCloud);

    if(noise){
        for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = outputCloud->begin(); it!= outputCloud->end(); it++){
            it->x = it->x + gsl_ran_gaussian(rng, noise);
            it->y = it->y + gsl_ran_gaussian(rng, noise);
            it->z = it->z + gsl_ran_gaussian(rng, noise);
        }
    }

    return outputCloud;
}

void particle_filter::resample_particles()
{
    WeightVector L = (weightVector - WeightVector::Constant(weightVector.maxCoeff())).array().exp();
    WeightVector Q = L / L.sum();
    WeightVector R = WeightVector::Zero();
    R(0,0) = Q(0,0);
    for(int i=1; i<NPOP_PARTICLES; i++)
        R(i,0) = R(i-1,0) + Q(i,0);

    srand((unsigned int) time(0));
    WeightVector T = WeightVector::Random();
    T = (T + WeightVector::Constant(1)) / 2;

    WeightVector I = WeightVector::Zero();
    for(int i=0; i<NPOP_PARTICLES; i++){
        for(int j=0; j<NPOP_PARTICLES; j++){
            if(R(j,0) >= T(i,0)){
                I(i,0) = j;
                break;
            }
        }
    }

    StateMatrix newStateMatrix = StateMatrix::Zero();
    for(int i=0; i<NPOP_PARTICLES; i++){
        newStateMatrix.col(i) = stateMatrix.col(I(i,0));
    }
    stateMatrix = newStateMatrix;
}
