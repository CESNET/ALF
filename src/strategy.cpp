#include <alf/strategy.hpp>

namespace alf {
    template<class MODEL>
    arma::uvec RandomStrategy<MODEL>::select(
            std::shared_ptr<MODEL> model,
            std::shared_ptr<arma::mat> labeled,
            std::shared_ptr<arma::mat> unlabeled) {
        if(m_count > unlabeled->n_cols) {
            return arma::randperm(unlabeled->n_cols, unlabeled->n_cols);
        }
        return arma::randperm(unlabeled->n_cols, m_count);
    }

    template<class MODEL>
    arma::uvec UncertaintyLCStrategy<MODEL>::select(
            std::shared_ptr<MODEL> model,
            std::shared_ptr<arma::mat> labeled,
            std::shared_ptr<arma::mat> unlabeled) {
        arma::cube cube_probabilities;
        model->Predict(*unlabeled, cube_probabilities);
        auto probabilities = cube_probabilities.slice(0);
        arma::vec max_probabilities = arma::max(probabilities, 1);
        arma::uvec indices = arma::sort_index(max_probabilities);
        return indices.subvec(0, m_count - 1);
    }

    template<class MODEL>
    arma::uvec UncertaintyEntropyStrategy<MODEL>::select(
            std::shared_ptr<MODEL> model,
            std::shared_ptr<arma::mat> labeled,
            std::shared_ptr<arma::mat> unlabeled) {
        arma::cube cube_probabilities;
        model->Predict(*unlabeled, cube_probabilities);
        auto probabilities = cube_probabilities.slice(0);
        arma::vec entropy = arma::sum(probabilities % arma::log(probabilities), 1);
        arma::uvec indices = arma::sort_index(entropy, "descend"); // we want highest entropy first
        return indices.subvec(0, m_count - 1);
    }

    template class RandomStrategy<RandomForestModel>;
    template class UncertaintyLCStrategy<RandomForestModel>;
    template class UncertaintyEntropyStrategy<RandomForestModel>;
    template class RandomStrategy<CommitteeModel>;
    template class UncertaintyLCStrategy<CommitteeModel>;
    template class UncertaintyEntropyStrategy<CommitteeModel>;
}
