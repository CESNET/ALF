#include <strategy.hpp>

namespace alf {

    arma::uvec RandomStrategy::select(
            std::shared_ptr<RandomForest<>> rf,
            std::shared_ptr<arma::mat> labeled,
            std::shared_ptr<arma::mat> unlabeled) {
        if(m_count > unlabeled->n_cols) {
            return arma::randperm(unlabeled->n_cols, unlabeled->n_cols);
        }
        return arma::randperm(unlabeled->n_cols, m_count);
    }

    arma::uvec UncertaintyLCStrategy::select(
            std::shared_ptr<RandomForest<>> rf,
            std::shared_ptr<arma::mat> labeled,
            std::shared_ptr<arma::mat> unlabeled) {
        arma::Row<size_t> predictions;
        arma::mat probabilities;
        rf->Classify(*unlabeled, predictions, probabilities);
        arma::vec max_probabilities = arma::max(probabilities, 1);
        arma::uvec indices = arma::sort_index(max_probabilities);
        return indices.subvec(0, 0);
    }

    arma::uvec UncertaintyEntropyStrategy::select(
            std::shared_ptr<RandomForest<>> rf,
            std::shared_ptr<arma::mat> labeled,
            std::shared_ptr<arma::mat> unlabeled) {
        arma::Row<size_t> predictions;
        arma::mat probabilities;
        rf->Classify(*unlabeled, predictions, probabilities);
        arma::vec entropy = arma::sum(probabilities % arma::log(probabilities), 1);
        arma::uvec indices = arma::sort_index(entropy, "descend"); // we want highest entropy first
        return indices.subvec(0, 0);
    }

}

