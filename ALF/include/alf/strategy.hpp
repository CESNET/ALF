#ifndef ALF_STRATEGY_HPP
#define ALF_STRATEGY_HPP

#include <memory>

#include <mlpack.hpp>

#include <alf/model.hpp>

using namespace mlpack;

namespace alf {
    template<class MODEL>
    class StrategyBase {
    public:
        StrategyBase() = default;
        ~StrategyBase() = default;
        virtual arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) = 0;
    };
    template<class MODEL>
    class RandomStrategy: public StrategyBase<MODEL> {
    public:
        explicit RandomStrategy(int count = 1): m_count(count) {};
        arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    private:
        int m_count;
    };
    template<class MODEL>
    class UncertaintyLCStrategy: public StrategyBase<MODEL> {
    public:
        explicit UncertaintyLCStrategy(int count = 1, double threshold = 0): m_count(count), m_threshold(threshold) {};
        arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    private:
        int m_count;
        double m_threshold;
    };
    template<class MODEL>
    class UncertaintyEntropyStrategy: public StrategyBase<MODEL> {
    public:
        explicit UncertaintyEntropyStrategy(int count = 1, double threshold = 0): m_count(count), m_threshold(threshold) {};
        arma::uvec select(
                std::shared_ptr<MODEL> model,
                std::shared_ptr<arma::mat> labeled,
                std::shared_ptr<arma::mat> unlabeled) override;
    private:
        int m_count;
        double m_threshold;
    };

}

#endif //ALF_STRATEGY_HPP
