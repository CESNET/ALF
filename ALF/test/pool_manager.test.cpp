#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <mlpack.hpp>
#include <fstream> // for copying files

#include <alf/state_manager.hpp>

using namespace alf;

TEST(PoolHandleTest, CreatingObjectPtr) {
	std::string path = "pathString";
	auto * poolHandler = new State_manager(path);
	EXPECT_TRUE(poolHandler != nullptr);
	delete poolHandler;
}

TEST(PoolHandleTest, LoadingLabeledDataset) {
	std::string path = TEST_RESOURCE_DIR "/test.db";
	auto * poolHandler = new State_manager(path);
	poolHandler->load_labeled();
	auto labeled = poolHandler->get_labeled();
	EXPECT_TRUE(labeled != nullptr);
	EXPECT_EQ(labeled->n_rows, 2);
	EXPECT_EQ(labeled->n_cols, 9);
	delete poolHandler;
}

TEST(PoolHandleTest, LoadingUnlabeledDataset) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    auto * poolHandler = new State_manager(path);
    poolHandler->load_unlabeled();
    auto unlabeled = poolHandler->get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 2);
    EXPECT_EQ(unlabeled->n_cols, 8);
    delete poolHandler;
}

TEST(PoolHandleTest, LoadingLabels) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    auto * poolHandler = new State_manager(path);
    poolHandler->load_labeled();
    auto labels = poolHandler->get_labels();
    EXPECT_TRUE(labels != nullptr);
    EXPECT_EQ(labels->n_elem, 9);
    delete poolHandler;
}

TEST(PoolHandleTest, GetLabelCount) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    auto * poolHandler = new State_manager(path);
    poolHandler->load_labeled();
    auto labelCount = poolHandler->get_labels_count();
    EXPECT_EQ(labelCount, 3);
    delete poolHandler;
}

// do not use original db file, copy it to a temporary file

TEST(PoolHandleTest, AnnotateUnlabeled) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    std::string tmpPath = TEST_RESOURCE_DIR "/test_tmp.db";
    std::ifstream src(path, std::ios::binary);
    std::ofstream dst(tmpPath, std::ios::binary);
    dst << src.rdbuf();
    src.close();
    dst.close();

    auto * poolHandler = new State_manager(tmpPath);
    poolHandler->load_unlabeled();
    auto unlabeled = poolHandler->get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 2);
    EXPECT_EQ(unlabeled->n_cols, 8);

    arma::uvec indices(2);
    indices[0] = 0;
    indices[1] = 2;
    poolHandler->annotate_unlabeled(indices);

    poolHandler->load_unlabeled();
    unlabeled = poolHandler->get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 2);
    EXPECT_EQ(unlabeled->n_cols, 6);
    // delete temporary file
    remove(tmpPath.c_str());
    delete poolHandler;
}

TEST(PoolHandleTest, AnnotateUnlabeled2) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    std::string tmpPath = TEST_RESOURCE_DIR "/test_tmp.db";
    std::ifstream src(path, std::ios::binary);
    std::ofstream dst(tmpPath, std::ios::binary);
    dst << src.rdbuf();
    src.close();
    dst.close();

    auto * poolHandler = new State_manager(tmpPath);
    poolHandler->load_unlabeled();
    auto unlabeled = poolHandler->get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 2);
    EXPECT_EQ(unlabeled->n_cols, 8);

    arma::uvec indices(2);
    indices[0] = 0;
    indices[1] = 2;
    poolHandler->annotate_unlabeled(indices);

    poolHandler->load_unlabeled();
    unlabeled = poolHandler->get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 2);
    EXPECT_EQ(unlabeled->n_cols, 6);

    indices[0] = 0;
    indices[1] = 1;
    poolHandler->annotate_unlabeled(indices);

    poolHandler->load_unlabeled();
    unlabeled = poolHandler->get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 2);
    EXPECT_EQ(unlabeled->n_cols, 4);

    // delete temporary file
    remove(tmpPath.c_str());
    delete poolHandler;
}