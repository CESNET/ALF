#include <alf/model.hpp>

namespace alf {

        void RandomForestModel::Train(arma::mat &labeled, arma::Row<size_t> &labels, size_t labels_count) {
            m_rf.Train(labeled, labels, labels_count);
        }

        void RandomForestModel::Predict(arma::mat &unlabeled, arma::cube &probabilities) {
            arma::Row<size_t> predictions;
            arma::mat probabilities_matrix;
            m_rf.Classify(unlabeled, predictions, probabilities_matrix);
            probabilities.set_size(probabilities_matrix.n_rows, probabilities_matrix.n_cols, 1);
            probabilities.slice(0) = probabilities_matrix;
        }

        void CommitteeModel::Train(arma::mat & labeled, arma::Row<size_t> & labels, size_t labels_count) {
            m_rf.Train(labeled, labels, labels_count);
            m_dt.Train(labeled, labels, labels_count);
            m_nb.Train(labeled, labels, labels_count);
        }

        void CommitteeModel::Predict(arma::mat & unlabeled, arma::cube &probabilities) {
            arma::Row<size_t> predictions;
            arma::mat rf_probabilities;
            arma::mat dt_probabilities;
            arma::mat nb_probabilities;
            m_rf.Classify(unlabeled, predictions, rf_probabilities);
            m_dt.Classify(unlabeled, predictions, dt_probabilities);
            m_nb.Classify(unlabeled, predictions, nb_probabilities);
            probabilities.set_size(rf_probabilities.n_rows, rf_probabilities.n_cols, 3);
            probabilities.slice(0) = rf_probabilities;
            probabilities.slice(1) = dt_probabilities;
            probabilities.slice(2) = nb_probabilities;
        }

}