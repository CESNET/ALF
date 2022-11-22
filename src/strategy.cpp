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
        MLResult mlResult;
        model->Predict(*unlabeled, mlResult);
        arma::mat probabilities = arma::mean(mlResult.probabilities, 2);
        arma::vec max_probabilities = arma::max(probabilities, 1);
        arma::uvec indices = arma::sort_index(max_probabilities); // TODO: optimalization, dont need to sort, just select top m_count, O(kn) where k << n is better than O(nlogn)
        return indices.subvec(0, m_count - 1);
    }

    template<class MODEL>
    arma::uvec UncertaintyEntropyStrategy<MODEL>::select(
            std::shared_ptr<MODEL> model,
            std::shared_ptr<arma::mat> labeled,
            std::shared_ptr<arma::mat> unlabeled) {
        MLResult mlResult;
        model->Predict(*unlabeled, mlResult);
		arma::mat probabilities = arma::mean(mlResult.probabilities, 2);
        arma::vec entropy = arma::sum(probabilities % arma::log(probabilities), 1);
        arma::uvec indices = arma::sort_index(entropy, "descend"); // TODO: we want highest entropy first, also optimalization is needed
        return indices.subvec(0, m_count - 1);
    }

	template<class MODEL>
	arma::uvec QBDStrategy<MODEL>::select(
			std::shared_ptr<MODEL> model,
			std::shared_ptr<arma::mat> labeled,
			std::shared_ptr<arma::mat> unlabeled) {
		if(model->PredictorCount() < 2) {
			throw std::runtime_error("Query-By-Disegreement requires at least 2 models in the model pool. Otherwise, this heuristic does not make sense.");
		}
		MLResult mlResult;
		model->Predict(*unlabeled, mlResult);
		std::vector<size_t> disagreement_indeces;
		for(size_t i = 0; i < mlResult.predictions.n_cols; ++i) {
			size_t comparer = mlResult.predictions.at(0, i);
			for (int j = 1; j < mlResult.predictions.n_rows; ++j) {
				if(mlResult.predictions.at(j, i) != comparer) {
					disagreement_indeces.emplace_back(i);
					break;
				}
			}
		}
		return arma::conv_to<arma::uvec>::from(disagreement_indeces);
	}

	template<class MODEL>
	arma::uvec QBCEntropyStrategy<MODEL>::select(
			std::shared_ptr<MODEL> model,
			std::shared_ptr<arma::mat> labeled,
			std::shared_ptr<arma::mat> unlabeled) {
		if(model->PredictorCount() < 2) {
			throw std::runtime_error("Query-By-Comitte-VoteEntropy requires at least 2 models in the model pool. Otherwise, this heuristic does not make sense.");
		}
		MLResult mlResult;
		model->Predict(*unlabeled, mlResult);
		auto committee_size = model->PredictorCount();
		auto entropies = arma::uvec(mlResult.predictions.n_cols, arma::fill::zeros);
		for(int c = 0; c < committee_size; ++c) {
			arma::Mat<size_t> tmp_matrix = mlResult.predictions.transform([c](size_t val) {
				return val == c ? 1 : 0;
			});
			for (int i = 0; i < mlResult.predictions.n_cols; ++i) {
				arma::Col<size_t> tmp_col = tmp_matrix.col(i);
				size_t vote_sum = arma::sum(tmp_col);
				entropies.at(i) += -(vote_sum/committee_size) * arma::log(vote_sum/committee_size);
			}
		}
		arma::uvec indices = arma::sort_index(entropies, "descend"); // TODO: we want highest entropy first, also optimalization is needed
		return indices.subvec(0, m_count - 1);
	}

	template<class MODEL>
	arma::uvec FarthestSampleStrategy<MODEL>::select(
			std::shared_ptr<MODEL> model,
			std::shared_ptr<arma::mat> labeled,
			std::shared_ptr<arma::mat> unlabeled) {
		auto labeled_cnt = labeled->n_cols;
		arma::uword random_index = arma::randi<arma::uword>(arma::distr_param(0,static_cast<int>(labeled_cnt - 1)));
		mlpack::DrusillaSelect<> drussila_select(*unlabeled, m_l, m_m);
		arma::mat distances;
		arma::Mat<size_t> neighbors;
		drussila_select.Search(*labeled, m_count, neighbors, distances);
		auto selection = neighbors.col(random_index);
		return arma::conv_to<arma::uvec>::from(selection);
	}

    template class RandomStrategy<RandomForestModel>;
    template class UncertaintyLCStrategy<RandomForestModel>;
    template class UncertaintyEntropyStrategy<RandomForestModel>;
    template class RandomStrategy<CommitteeModel>;
    template class UncertaintyLCStrategy<CommitteeModel>;
    template class UncertaintyEntropyStrategy<CommitteeModel>;
}

