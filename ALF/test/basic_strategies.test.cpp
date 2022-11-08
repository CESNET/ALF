#include <gtest/gtest.h>

#include <string>
#include <iostream>

#include <mlpack.hpp>

#include <alf/state_manager.hpp>
#include <alf/strategy.hpp>
#include <alf/model.hpp>

using namespace alf;

TEST(BasicStrategies, CreatingObjectPtr) {
    std::string path = "pathString";
    auto * poolHandler = new State_manager(path);
    EXPECT_TRUE(poolHandler != nullptr);
    delete poolHandler;
}

TEST(BasicStrategies, RandomStrategy) {
    auto data = std::make_shared<arma::mat>("0.339406815,0.843176636,0.472701471; \
                  0.160147626,0.255047893,0.04072469;  \
                  0.564535197,0.943435462,0.597070812");
    auto model = std::make_shared<RandomForestModel>();

    auto strategy = RandomStrategy<RandomForestModel>(1);
    auto result = strategy.select(model, data, data);
    EXPECT_TRUE((0 <= result[0]) && (result[0] < 4));
    EXPECT_TRUE(result.n_elem == 1);
}

TEST(BasicStrategies, RandomStrategySelectMoreThanAvailable) {
    auto data = std::make_shared<arma::mat>("0.339406815,0.843176636,0.472701471; \
                  0.160147626,0.255047893,0.04072469;  \
                  0.564535197,0.943435462,0.597070812");
    auto model = std::make_shared<RandomForestModel>();

    auto strategy = RandomStrategy<RandomForestModel>(4);
    auto result = strategy.select(model, data, data);
    EXPECT_TRUE((0 <= result[0]) && (result[0] < 4));
    EXPECT_TRUE(result.n_elem == 3);
}

TEST(BasicStrategies, UncertaintyLCStrategy) {
    auto labeled = std::make_shared<arma::mat>("1,0,0; \
                  0,1,0;  \
                  0,0,1");
    auto labels = std::make_shared<arma::Row<size_t>>("0,1,2");
    auto model = std::make_shared<RandomForestModel>();
    model->Train(*labeled, *labels, 3);

    auto strategy = UncertaintyLCStrategy<RandomForestModel>();
    auto result = strategy.select(model, labeled, labeled);
    EXPECT_TRUE(result.n_elem == 1);
    EXPECT_TRUE(result[0] == 2);
}

TEST(BasicStrategies, UncertaintyEntropyStrategy) {
    auto labeled = std::make_shared<arma::mat>("1,0,0; \
                  0,1,0;  \
                  0,0,1");
    auto labels = std::make_shared<arma::Row<size_t>>("0,1,2");
    auto model = std::make_shared<RandomForestModel>();
    model->Train(*labeled, *labels, 3);

    auto strategy = UncertaintyEntropyStrategy<RandomForestModel>();
    auto result = strategy.select(model, labeled, labeled);
    EXPECT_TRUE(result.n_elem == 1);
    EXPECT_TRUE(result[0] == 2);
}
