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
        arma::uvec operator()(
                std::shared_ptr<RandomForest<>> rf,
                std::shared_ptr<arma::mat> &labeled,
                std::shared_ptr<arma::mat> &unlabeled);
    };

    class RandomStrategy: public StrategyBase {};

}

#endif //ALF_STRATEGY_HPP
