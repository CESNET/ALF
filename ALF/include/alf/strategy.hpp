#ifndef ALF_STRATEGY_HPP
#define ALF_STRATEGY_HPP

#include <memory>

#include <mlpack.hpp>

using namespace mlpack;

namespace alf {
    class StrategyBase {
    public:
        StrategyBase() = default;
        ~StrategyBase() = default;
        virtual arma::uvec select(
                std::shared_ptr<RandomForest<>> rf,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) = 0;
    };

    class RandomStrategy: public StrategyBase {
    public:
        explicit RandomStrategy(int count = 1): m_count(count) {};
        arma::uvec select(
                std::shared_ptr<RandomForest<>> rf,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    private:
        int m_count;
    };

    class UncertaintyLCStrategy: public StrategyBase {
    public:
        arma::uvec select(
                std::shared_ptr<RandomForest<>> rf,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    };

    class UncertaintyEntropyStrategy: public StrategyBase {
    public:
        arma::uvec select(
                std::shared_ptr<RandomForest<>> rf,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    };

}

#endif //ALF_STRATEGY_HPP
