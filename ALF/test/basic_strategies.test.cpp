#include <gtest/gtest.h>

#include <string>
#include <iostream>

#include <mlpack.hpp>

#include <alf/state_manager.hpp>
#include <alf/strategy.hpp>

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
    auto rf = std::make_shared<mlpack::RandomForest<>>();

    auto strategy = RandomStrategy(1);
    auto result = strategy.select(rf, data, data);
    EXPECT_TRUE((0 <= result[0]) && (result[0] < 4));
    EXPECT_TRUE(result.n_elem == 1);
}

TEST(BasicStrategies, RandomStrategySelectMoreThanAvailable) {
    auto data = std::make_shared<arma::mat>("0.339406815,0.843176636,0.472701471; \
                  0.160147626,0.255047893,0.04072469;  \
                  0.564535197,0.943435462,0.597070812");
    auto rf = std::make_shared<mlpack::RandomForest<>>();

    auto strategy = RandomStrategy(4);
    auto result = strategy.select(rf, data, data);
    EXPECT_TRUE((0 <= result[0]) && (result[0] < 4));
    EXPECT_TRUE(result.n_elem == 3);
}

TEST(BasicStrategies, UncertaintyLCStrategy) {
    auto labeled = std::make_shared<arma::mat>("1,0,0; \
                  0,1,0;  \
                  0,0,1");
    auto labels = std::make_shared<arma::Row<size_t>>("0,1,2");
    auto rf = std::make_shared<mlpack::RandomForest<>>();
    rf ->Train(*labeled, *labels, 3);

    auto strategy = UncertaintyLCStrategy();
    auto result = strategy.select(rf, labeled, labeled);
    EXPECT_TRUE(result.n_elem == 1);
    EXPECT_TRUE(result[0] == 2);
}

TEST(BasicStrategies, UncertaintyEntropyStrategy) {
    auto labeled = std::make_shared<arma::mat>("1,0,0; \
                  0,1,0;  \
                  0,0,1");
    auto labels = std::make_shared<arma::Row<size_t>>("0,1,2");
    auto rf = std::make_shared<mlpack::RandomForest<>>();
    rf ->Train(*labeled, *labels, 3);

    auto strategy = UncertaintyEntropyStrategy();
    auto result = strategy.select(rf, labeled, labeled);
    EXPECT_TRUE(result.n_elem == 1);
    EXPECT_TRUE(result[0] == 2);
}


