#include <alf/model.hpp>

namespace alf {

        void RandomForestModel::Train(arma::mat &labeled, arma::Row<size_t> &labels, size_t labels_count) {
            m_rf.Train(labeled, labels, labels_count);
        }

        void RandomForestModel::Predict(arma::mat &unlabeled, arma::mat &probabilities) {
            arma::Row<size_t> predictions;
            m_rf.Classify(unlabeled, predictions, probabilities);
        }

}