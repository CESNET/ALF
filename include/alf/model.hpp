#ifndef ALF_MODEL_HPP
#define ALF_MODEL_HPP

#include <memory>
#include <mlpack.hpp>

namespace alf {
    class IModel {
    public:
        virtual void Train(arma::mat &labeled, arma::Row<size_t> &labels, size_t labels_count) = 0;
        virtual void Predict(arma::mat &unlabeled, arma::cube &probabilities) = 0;
        virtual int PredictorCount() = 0;
    };

    class RandomForestModel: public IModel {
    public:
        RandomForestModel() = default;
        ~RandomForestModel() = default;
        void Train(arma::mat & labeled, arma::Row<size_t> & labels, size_t labels_count) override;
        void Predict(arma::mat & unlabeled, arma::cube &probabilities) override;
        int PredictorCount() override { return 1; }
    private:
        mlpack::RandomForest<> m_rf;
    };

    class CommitteeModel: public IModel {
    public:
        CommitteeModel() = default;
        ~CommitteeModel() = default;
        void Train(arma::mat & labeled, arma::Row<size_t> & labels, size_t labels_count) override;
        void Predict(arma::mat & unlabeled, arma::cube &probabilities) override;
        int PredictorCount() override { return 3; }
    private:
        mlpack::RandomForest<> m_rf;
        mlpack::DecisionTree<> m_dt;
        mlpack::NaiveBayesClassifier<> m_nb;
    };






}

#endif //ALF_MODEL_HPP
