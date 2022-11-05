
#include <strategy.hpp>

namespace alf {

    arma::uvec StrategyBase::operator()(
            std::shared_ptr<RandomForest<>>  rf,
            std::shared_ptr<arma::mat> &labeled,
            std::shared_ptr<arma::mat> &unlabeled) {
        return arma::uvec({1,2,3});
    }

}

