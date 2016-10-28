#include "FeedforwardNeuralNetwork.h"
#include "common/util.h"
#include "common/TrainSettings.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
#include <pthread.h>

using std::runtime_error;
using std::cout;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct WorkerParams {
    const Eigen::MatrixXd* x;
    const Eigen::MatrixXd* y;
    //int first, last;
    const TrainSettings* train_settings;
    std::vector<MatrixXd>* weights;
};

FeedforwardNeuralNetwork::FeedforwardNeuralNetwork(const std::vector<int>& layers) : layers(layers) {
    if (layers.size() < 2) {
        throw runtime_error("Neural networks need at least an input layer and an output layer");
    }

    if (!std::all_of(layers.begin(), layers.end(), [](int element) { return element > 0; })) {
        throw runtime_error("All layers must have a positive number of units");
    }

    for (size_t i = 0; i < layers.size() - 1; ++i) {
        weight_list.push_back(MatrixXd::Zero(layers[i + 1], layers[i] + 1));
    }
}

void FeedforwardNeuralNetwork::back_propagation(const std::vector<Eigen::VectorXd> &fp_results,
                                                const Eigen::VectorXd &y,
                                                const std::vector<Eigen::MatrixXd> &weight_list,
                                                std::vector<Eigen::MatrixXd> &out) {
    VectorXd delta_next = fp_results[fp_results.size() - 1] - y;

    for (int i = (int)fp_results.size() - 2; i >= 0; --i) {
        out[i] += delta_next * fp_results[i].transpose();
        delta_next = (weight_list[i].transpose() * delta_next).cwiseProduct(fp_results[i])
                .cwiseProduct(fp_results[i].unaryExpr([](double element) { return 1 - element; }));
        delta_next = delta_next.bottomRows(delta_next.rows() - 1);
    }
}

void FeedforwardNeuralNetwork::back_propagation(const std::vector<Eigen::VectorXd>& fp_results,
                                                const Eigen::VectorXd& y,
                                                std::vector<Eigen::MatrixXd>& out) {
    back_propagation(fp_results, y, weight_list, out);
}

std::vector<Eigen::VectorXd> FeedforwardNeuralNetwork::forward_propagation(const Eigen::VectorXd &input,
                                                                           const std::vector<Eigen::MatrixXd> &weight_list) {
    if (input.rows() != weight_list[0].cols() - 1){
        std::cout << "Input rows: " << input.rows() << ". Expected: " << (weight_list[0].cols() - 1) <<'\n';
        throw runtime_error("Invalid input size");
    }

    std::vector<VectorXd> result(weight_list.size() + 1);

    VectorXd extended_vector(input.rows() + 1);
    extended_vector << 1, input;
    result[0] = extended_vector;

    for (int i = 0; i < (int)weight_list.size() - 1; ++i) {
        extended_vector = VectorXd(weight_list[i].rows() + 1);
        extended_vector << 1, (weight_list[i] * result[i]).unaryExpr(&sigmoid);
        result[i + 1] = extended_vector;
    }

    int last = (int)result.size() - 1;
    result[last] = (weight_list[last - 1] * result[last - 1]).unaryExpr(&sigmoid);

    return result;
}

std::vector<Eigen::VectorXd> FeedforwardNeuralNetwork::forward_propagation(const Eigen::VectorXd &input) {
    return forward_propagation(input, this->weight_list);
}

void FeedforwardNeuralNetwork::set_weights(int source_layer, const Eigen::MatrixXd &weights) {
    if (source_layer < 0 || source_layer >= static_cast<int>(weight_list.size())) {
        throw runtime_error("Invalid layer index");
    }

    if (weights.rows() != weight_list[source_layer].rows() || weights.cols() != weight_list[source_layer].cols()) {
        throw runtime_error("Invalid weight matrix dimension");
    }

    weight_list[source_layer] = weights;
}

Eigen::VectorXd FeedforwardNeuralNetwork::predict(const Eigen::VectorXd &input) {
    std::vector<Eigen::VectorXd> all_results = forward_propagation(input);

    return all_results[all_results.size() - 1];
}

Eigen::MatrixXd FeedforwardNeuralNetwork::random_matrix(int rows, int cols, double epsilon, std::mt19937& generator) {
  std::uniform_real_distribution<> distribution(-epsilon, epsilon);
  MatrixXd result(rows, cols);

  for(int i = 0; i < rows; ++i) {
    for(int j = 0; j < cols; ++j) {
      result(i, j) = distribution(generator);
    }
  }

  return result;
}

TrainResult FeedforwardNeuralNetwork::train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                                            const TrainSettings& train_settings) {
    if (x.rows() != y.rows()) {
        throw runtime_error("Invalid training data");
    }

    train_settings.validate();

    // Randomly initialize vectors
    if (train_settings.initialize_weights_randomly) {
      for(size_t i = 0; i < weight_list.size(); ++i) {
          set_weights(i, FeedforwardNeuralNetwork::random_matrix(
            weight_list[i].rows(), weight_list[i].cols(), train_settings.random_epsilon,
            *train_settings.generator));
      }
    }

    int threads = train_settings.threads;

    std::cout << "Threads: " << threads << std::endl;

    double error = compute_error(x, y, train_settings.regularization_term);

    // Gradient-descent
    for(int it = 1; it <= train_settings.iterations; ++it) {
        if (error <= train_settings.target_error) {
          return TrainResult(it, error);
        }

        // Iterate over each example. Divide the examples in groups for parallel processing
        //int block_size = static_cast<int>(ceil((1.0 * x.rows()) / threads));
        std::vector<pthread_t> thread_ids(threads);
        std::vector<std::vector<MatrixXd>> worker_weights(threads);
        std::vector<WorkerParams> param_list(threads);
        for (int i = 0; i < threads; ++i) {
            worker_weights[i] = weight_list;

            WorkerParams &params = param_list[i];
            params.weights = &worker_weights[i];
            params.x = &x;
            params.y = &y;
            params.train_settings = &train_settings;

            if (pthread_create(&thread_ids[i], NULL, &FeedforwardNeuralNetwork::do_gradient_descent, &params)) {
                throw runtime_error("Unable to create thread");
            }
        }

        for (size_t i = 0; i < weight_list.size(); ++i) {
            weight_list[i] = MatrixXd::Zero(weight_list[i].rows(), weight_list[i].cols());
        }

        for(int i = 0; i < threads; ++i) {
            if (pthread_join(thread_ids[i], NULL)) {
                throw runtime_error("Unable to join thread");
            }

            for (size_t j = 0; j < weight_list.size(); ++j) {
                weight_list[j] += worker_weights[i][j];
            }
        }

        for (size_t i = 0; i < weight_list.size(); ++i) {
            weight_list[i] *= 1.0 / threads;
        }

        error = compute_error(x, y, train_settings.regularization_term);
    }

    return TrainResult(train_settings.iterations, error);
}

void* FeedforwardNeuralNetwork::do_gradient_descent(void *params_unsafe) {
    struct WorkerParams* params = (WorkerParams*)params_unsafe;

    std::vector<MatrixXd> &weights = *params->weights;
    std::vector<MatrixXd> deltas(weights.size()); // layers - 1
    std::vector<MatrixXd> gradients(weights.size());

    for(size_t i = 0; i < gradients.size(); ++i) {
        gradients[i] = MatrixXd::Zero(weights[i].rows(), weights[i].cols());
    }

    //int steps = params->train_settings->inner_steps / params->train_settings->threads;
    int steps = params->train_settings->inner_steps;
    //double cost = compute_error(*params->x, *params->y, params->train_settings->regularization_term, weights);
    //std::cout << "Initial cost: " << cost << "\n";

    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<int> distribution(0, params->x->rows() - 1);

    for (int t = 0; t < steps; ++t) {

        for (size_t k = 0; k < deltas.size(); ++k) {
            deltas[k] = MatrixXd::Zero(weights[k].rows(), weights[k].cols());
        }

        int example = distribution(generator);
        VectorXd xi = params->x->row(example).transpose();
        VectorXd yi = params->y->row(example).transpose();

        std::vector<Eigen::VectorXd> fp_results = forward_propagation(xi, weights);
        back_propagation(fp_results, yi, weights, deltas);

        /* Check code BEGIN */
        //MatrixXd numerical_gradient = compute_numerical_gradient(*params);
        /* Check code END */

        for (size_t i = 0; i < deltas.size(); ++i) {
            /*MatrixXd current = (1.0 / params->x->rows()) * deltas[i];

            current.rightCols(current.cols() - 1) +=
                    (params->train_settings->regularization_term / params->x->rows()) *
                    weights[i].rightCols(weights[i].cols() - 1);*/



            //gradients[i] = params->train_settings->momentum * gradients[i] + current;
            gradients[i] = params->train_settings->momentum * gradients[i] + (1.0 / params->x->rows()) * deltas[i];
            gradients[i].rightCols(gradients[i].cols() - 1) += (params->train_settings->regularization_term / params->x->rows()) * weights[i].rightCols(weights[i].cols() - 1);
            weights[i] -= params->train_settings->step_factor * gradients[i];
        }

        /*double new_cost = compute_error(*params->x, *params->y, params->train_settings->regularization_term, weights);

        //std::cout << "New cost:" << new_cost << ", delta: " << (new_cost - cost) << "\n";

        cost = new_cost;*/
    }

    return 0;
}

double FeedforwardNeuralNetwork::compute_error(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, double regularization_term,
  const std::vector<Eigen::MatrixXd>& weight_list) {
    if (x.rows() != y.rows()) {
        throw runtime_error("Different number of x and y examples");
    }

    if (y.cols() != weight_list[weight_list.size() - 1].rows()) {
        throw runtime_error("Unexpected number of output classes");
    }

    double error = 0;
    for(int i = 0; i < x.rows(); ++i) {
        std::vector<Eigen::VectorXd> all_results = forward_propagation((x.row(i).transpose()), weight_list);

        auto prediction = all_results[all_results.size() - 1];

        error += pow((y.row(i).transpose() - prediction).norm(), 2);

        /*for (int j = 0; j < prediction.size(); ++j) {
            double y_j = y(i, j);
            double h_j = prediction(j, 0);
            error += y_j * log(h_j) + (1 - y_j) * log(1 - h_j);
        }*/
    }

    double complexity_term = 0;
    for (size_t i = 0; i < weight_list.size(); ++i) {
        for (int j = 0; j < weight_list[i].rows(); ++j) {
            for (int k = 0; k < weight_list[i].cols(); ++k) {
                complexity_term += weight_list[i](j, k) * weight_list[i](j, k);
            }
        }
    }

    //return -1.0 / x.rows() * error + regularization_term / (2 * x.rows()) * complexity_term;
    return 1.0 / x.rows() * error + regularization_term / (2 * x.rows()) * complexity_term;
}

double FeedforwardNeuralNetwork::compute_error(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                                               double regularization_term) {
    return compute_error(x, y, regularization_term, this->weight_list);
}
