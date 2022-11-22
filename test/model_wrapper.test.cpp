#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <mlpack.hpp>

#include <alf/state_manager.hpp>
#include <alf/model.hpp>

using namespace alf;

/* TODO: why segfaulting on github?
TEST(ModelWrapperTest, RandomForrestTest) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    auto poolHandler = State_manager(path);
    poolHandler.load_labeled();
    auto labeled = poolHandler.get_labeled();
    auto labels = poolHandler.get_labels();
    auto labels_count = poolHandler.get_labels_count();
    RandomForestModel model;
    model.Train(*labeled, *labels, labels_count);
    poolHandler.load_unlabeled();
    auto unlabeled = poolHandler.get_unlabeled();
    arma::cube probabilities;
    model.Predict(*unlabeled, probabilities);
    EXPECT_EQ(probabilities.n_rows, labels_count);
    EXPECT_EQ(probabilities.n_cols, unlabeled->n_cols);
    EXPECT_EQ(probabilities.n_slices, model.PredictorCount());
}

TEST(ModelWrapperTest, CommitteeModelTest) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    auto poolHandler = State_manager(path);
    poolHandler.load_labeled();
    auto labeled = poolHandler.get_labeled();
    auto labels = poolHandler.get_labels();
    auto labels_count = poolHandler.get_labels_count();
    RandomForestModel model;
    model.Train(*labeled, *labels, labels_count);
    poolHandler.load_unlabeled();
    auto unlabeled = poolHandler.get_unlabeled();
    arma::cube probabilities;
    model.Predict(*unlabeled, probabilities);
    EXPECT_EQ(probabilities.n_rows, labels_count);
    EXPECT_EQ(probabilities.n_cols, unlabeled->n_cols);
    EXPECT_EQ(probabilities.n_slices, model.PredictorCount());
}
*/