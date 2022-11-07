#ifndef ALF_MODEL_HPP
#define ALF_MODEL_HPP

#include <memory>
#include <mlpack.hpp>

namespace alf {
    class IModel {
    public:
        virtual void Train(arma::mat &labeled, arma::Row<size_t> &labels, size_t labels_count) = 0;
        virtual void Predict(arma::mat &unlabeled, arma::mat &probabilities) = 0;
    };

    class RandomForestModel: public IModel {
    public:
        RandomForestModel() = default;
        ~RandomForestModel() = default;
        void Train(arma::mat & labeled, arma::Row<size_t> & labels, size_t labels_count) override;
        void Predict(arma::mat & unlabeled, arma::mat &probabilities) override;
    private:
        mlpack::RandomForest<> m_rf;
    };






}

#endif //ALF_MODEL_HPP
