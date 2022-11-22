#include <alf/model.hpp>

namespace alf {

        void RandomForestModel::Train(arma::mat &labeled, arma::Row<size_t> &labels, size_t labels_count) {
            m_rf.Train(labeled, labels, labels_count);
        }

        void RandomForestModel::Predict(arma::mat &unlabeled, alf::MLResult &mlResult) {
			arma::Row<size_t> predictions;
            arma::mat probabilities_matrix;
            m_rf.Classify(unlabeled, predictions, probabilities_matrix);
			mlResult.append(probabilities_matrix, predictions);
        }

        void CommitteeModel::Train(arma::mat & labeled, arma::Row<size_t> & labels, size_t labels_count) {
            m_rf.Train(labeled, labels, labels_count);
            m_dt.Train(labeled, labels, labels_count);
            m_nb.Train(labeled, labels, labels_count);
        }

        void CommitteeModel::Predict(arma::mat & unlabeled, alf::MLResult &mlResult) {
            arma::Row<size_t> predictions;
            arma::mat probabilities;
			// TODO: ugly; not sure how to do it better for now
            m_rf.Classify(unlabeled, predictions, probabilities);
			mlResult.append(probabilities, predictions);
            m_dt.Classify(unlabeled, predictions, probabilities);
			mlResult.append(probabilities, predictions);
            m_nb.Classify(unlabeled, predictions, probabilities);
			mlResult.append(probabilities, predictions);
        }

}