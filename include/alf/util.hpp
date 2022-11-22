#ifndef ALF_UTIL_HPP
#define ALF_UTIL_HPP

#include <mlpack.hpp>
#include <ctime>

namespace alf {
	class MLResult {
	public:
		arma::Mat<size_t> predictions;
		arma::cube probabilities;
		void set_problem_dimension(size_t rows, size_t cols, size_t predictors) {
			predictions.set_size(rows, cols);
			probabilities.set_size(rows, cols, predictors);
		}
		void append(arma::mat &input_probabilities, arma::Row<size_t> &input_predictions) {
			probabilities.slice(i) = input_probabilities;
			predictions.row(i) = input_predictions;
			i++;
		}
	private:
		int i = 0;
	};

	struct FlowLabeledRecord {
		int flow_id;
		int matrix_index;
		int label;
		std::time_t timestamp;
		int predicted_label;
		double predicted_proba;
	};

	struct FlowUnlabeledRecord {
		int flow_id;
		int matrix_index;
	};
}

#endif // ALF_UTIL_HPP
