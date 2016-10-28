
#ifndef NEURAL_NETWORKS_UTIL_H
#define NEURAL_NETWORKS_UTIL_H

#include <string>
#include <Eigen/Dense>
#include <random>
#include <utility>

double sigmoid(double t);

std::string get_matrix_string(const Eigen::MatrixXd& matrix);

Eigen::MatrixXd generate_data(int n, std::mt19937& generator);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> read_mnist_database(const std::string& image_file_name, const std::string& label_file_name);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> read_iris_database(const std::string& file_name);

#endif //NEURAL_NETWORKS_UTIL_H
