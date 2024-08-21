#ifndef ADAPTIVE_KDE_H
#define ADAPTIVE_KDE_H

#include <vector>
#include <string>

// PointCloud structure
template <typename T>
struct PointCloud
{
    std::vector<std::vector<T> > pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return pts[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

double kernelDensityEstimation(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double epsilon, double h);
std::vector<double> kdProjection(const std::vector<std::vector<double>>& X, const std::vector<double>& z);
double gaussianKernel(const std::vector<double>& u, double h);
double gaussianKernel(const double dist, double h);
double gaussianKernelSquared(const std::vector<double>& u, double h);
double computeGaussianKernel(const std::vector<double>& x, const std::vector<double>& z, double h);
double estimateKernelSquared(const PointCloud<double>& cloud, const std::vector<double>& query, double h);
double adaptiveKDE(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double variance_estimate, double epsilon, double h);
double trueKernelDensity(const std::vector<std::vector<double>>& X, const std::vector<double>& z, double h);
std::vector<std::vector<double>> readDataset(const std::string& filename);

#endif // ADAPTIVE_KDE_H