#include "adaptshell_kde.h"
#include "nanoflann.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>

using namespace nanoflann;

double kernelDensityEstimation(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double epsilon, double h) {
    PointCloud<double> cloud;
    cloud.pts = X;
    double variance_estimate = estimateKernelSquared(cloud, z, h);
    return adaptiveKDE(X, z, variance_estimate, epsilon, h);
}

std::vector<double> kdProjection(const std::vector<std::vector<double>>& X, const std::vector<double>& z) {
    std::vector<double> projections;
    projections.reserve(X.size());
    
    for (const auto& x : X) {
        double projection = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            projection += (x[i] - z[i]) * (x[i] - z[i]);
        }
        projections.push_back(std::sqrt(projection));
    }
    
    return projections;
}

double gaussianKernel(const std::vector<double>& u, double h) {
    double squared_norm = 0.0;
    for (double ui : u) {
        squared_norm += ui * ui;
    }
    return std::exp(-squared_norm / (2 * h * h));
}

double gaussianKernel(const double dist, double h) {
    return std::exp(-(dist * dist) / (2 * h * h));
}

double gaussianKernelSquared(const std::vector<double>& u, double h) {
    double k = gaussianKernel(u, h);
    return k * k;
}

double computeGaussianKernel(const std::vector<double>& x, const std::vector<double>& z, double h) {
    double squared_norm = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        squared_norm += (x[i] - z[i]) * (x[i] - z[i]);
    }
    return gaussianKernel(squared_norm, h);
}

double estimateKernelSquared(const PointCloud<double>& cloud, const std::vector<double>& query, double h) {
    typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double> >, PointCloud<double>, -1> my_kd_tree_t;

    int dim = query.size();
    my_kd_tree_t index(dim, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    size_t dataset_size = cloud.pts.size();

    double closest_distance = 0.0;
    size_t k = 1;
    while (closest_distance == 0.0 && k <= dataset_size) {
        std::vector<size_t> ret_index(k);
        std::vector<double> out_dist_sqr(k);
        nanoflann::KNNResultSet<double> resultSet(k);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        index.findNeighbors(resultSet, &query[0], nanoflann::SearchParameters());
        
        for (size_t i = 0; i < k; ++i) {
            if (out_dist_sqr[i] > 0.0) {
                closest_distance = std::sqrt(out_dist_sqr[i]);
                break;
            }
        }
        
        if (closest_distance == 0.0) {
            k++;
        }
    }

    if (closest_distance == 0.0) {
        std::cout << "Warning: Could not find a non-zero distance." << std::endl;
        return 0.0;
    }

    double kernel_sq_estimate = 0;
    size_t points_counted = 0;
    double current_radius = closest_distance;
    size_t total_points = 0;
    int iters = 0;
    double closest_radius = current_radius;

    while (total_points < dataset_size) {
        std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
        nanoflann::SearchParameters params;
        size_t count = index.radiusSearch(&query[0], current_radius*current_radius, ret_matches, params);
        size_t new_points = count - total_points;
        if (new_points > 0) {
            double k = gaussianKernel(closest_radius, h);
            double k_sq = k * k;
            kernel_sq_estimate += new_points * k_sq;
            points_counted += new_points;
        }
        
        current_radius *= 2;
        total_points = count;
        if (iters > 0) {
            closest_radius *= 2;
        }

        iters++;
    }

    std::cout << "variance estimate: " << kernel_sq_estimate / points_counted << ", points_counted: " << points_counted << std::endl;

    return points_counted > 0 ? kernel_sq_estimate / points_counted : 0;
}

double adaptiveKDE(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double variance_estimate, double epsilon, double h) {
    double n = X.size();
    double d = z.size();  

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, n - 1);

    int j = 1;
    int r = 0;
    double average = 0;
    double kernel_sum;
    std::vector<double> diff(d);
    int reps = 0;
    while (average == 0 && reps < n) {
        r = distribution(generator);
        kernel_sum = computeGaussianKernel(X[r], z, h);
        if(reps % 1000 == 0) {
        }
        reps++;
        average = kernel_sum;
    }
    if(average == 0) {
        std::cout << "Error: Could not find a non-zero kernel sum." << std::endl;
        return 0.0;
    }
    std::cout << "Initial kernel value: " << average << std::endl;
        
    double t = 2 * variance_estimate / (std::pow(average, 2) * std::pow(epsilon, 2));

    std::cout << "Initial T: " << t << std::endl;
    while (j <= t && j <= n) {
        r = distribution(generator);
        kernel_sum += computeGaussianKernel(X[r], z, h);
        j++;
        average = kernel_sum / j;
        t = 2 * variance_estimate / (std::pow(epsilon, 2) * std::pow(average, 2));
        if (j % int(n / 10) == 0) {
            std::cout << "j: " << j << ", t: " << t << std::endl;
        }
    }

    std::cout << "Final kernel estimate: " << average << std::endl;
    std::cout << "Number of samples (j): " << j << std::endl;

    return average;
}

double trueKernelDensity(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double h) {
    double n = X.size();
    double d = z.size();
    
    double density = 0.0;
    std::vector<double> diff(d);
    for (const auto& x : X) {
        density += computeGaussianKernel(x, z, h);
    }
    
    std::cout << "True kernel density: " << density / n << std::endl;
    return density / n;
}

std::vector<std::vector<double>> readDataset(const std::string& filename) {
    std::vector<std::vector<double>> dataset;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        dataset.push_back(row);
    }
    
    return dataset;
}

int main() {
    std::string filename = "Skin_NonSkin.txt";
    std::vector<std::vector<double>> dataset = readDataset(filename);
    
    if (dataset.empty()) {
        std::cerr << "Failed to read the dataset or the dataset is empty." << std::endl;
        return 1;
    }
    
    // select random query and remove from dataset
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, dataset.size() - 1);
    int randomIndex = distribution(generator);
    std::vector<double> query = dataset[randomIndex];
    dataset.erase(dataset.begin() + randomIndex);
    
    // params: epsilon (error rate), h (bandwidth for gaussian kernel)
    double epsilon = 0.1;  
    double h = std::pow(dataset.size(), -1.0 / (dataset[0].size() + 4)); 
    std::cout << "Bandwidth (h): " << h << std::endl;

    h *= 10;
    
    double adaptiveDensity = kernelDensityEstimation(dataset, query, epsilon, h);
    double trueDensity = trueKernelDensity(dataset, query, h);
    double percentError = (trueDensity - adaptiveDensity) / trueDensity * 100;
    
    // Output results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Kernel Density Estimation results:" << std::endl;
    std::cout << "Query point index: " << randomIndex << std::endl;
    std::cout << "Number of samples: " << dataset.size() << std::endl;
    std::cout << "Percent error: " << percentError << "%" << std::endl;
    
    return 0;
}