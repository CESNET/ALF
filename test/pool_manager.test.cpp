#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <mlpack.hpp>

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
	EXPECT_EQ(labeled->n_rows, 20);
	EXPECT_EQ(labeled->n_cols, 1036);
    auto labels = poolHandler->get_labels();
    EXPECT_TRUE(labels != nullptr);
    EXPECT_EQ(labels->n_elem, 1036);
    delete poolHandler;
}


TEST(PoolHandleTest, LoadingDbManyTimesInOneObject) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    auto poolHandler = State_manager(path);
    for (int i = 0; i < 5; ++i){
        poolHandler.load_unlabeled();
    }
}

TEST(PoolHandleTest, AnnotateUnlabeled) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    std::string tmpPath = TEST_RESOURCE_DIR "/test_tmp.db";
    std::ifstream src(path, std::ios::binary);
    std::ofstream dst(tmpPath, std::ios::binary);
    dst << src.rdbuf();
    src.close();
    dst.close();

    auto poolHandler = State_manager(tmpPath);
    poolHandler.load_unlabeled();
    auto unlabeled = poolHandler.get_unlabeled();


    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 20);
    EXPECT_EQ(unlabeled->n_cols, 2000);

    arma::uvec indices(2);
    indices[0] = 0;
    indices[1] = 1;
    poolHandler.annotate_unlabeled(indices);

    poolHandler.load_unlabeled();
    unlabeled = poolHandler.get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 20);
    EXPECT_EQ(unlabeled->n_cols, 1998);
    // delete temporary file
    remove(tmpPath.c_str());
}

TEST(PoolHandleTest, AnnotateUnlabeledManyInLoop) {
    std::string path = TEST_RESOURCE_DIR "/test.db";
    std::string tmpPath = TEST_RESOURCE_DIR "/test_tmp.db";
    std::ifstream src(path, std::ios::binary);
    std::ofstream dst(tmpPath, std::ios::binary);
    dst << src.rdbuf();
    src.close();
    dst.close();

    auto poolHandler = State_manager(tmpPath);
    poolHandler.load_unlabeled();
    auto unlabeled = poolHandler.get_unlabeled();


    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 20);
    EXPECT_EQ(unlabeled->n_cols, 2000);

    for( int i = 0 ; i < 5; ++i){
        poolHandler.load_unlabeled();
        arma::uvec indices(2);
        indices[0] = 0;
        indices[1] = 1;
        poolHandler.annotate_unlabeled(indices);
    }
    poolHandler.load_unlabeled();
    unlabeled = poolHandler.get_unlabeled();
    EXPECT_TRUE(unlabeled != nullptr);
    EXPECT_EQ(unlabeled->n_rows, 20);
    EXPECT_EQ(unlabeled->n_cols, 1990);
    // delete temporary file
    remove(tmpPath.c_str());
}
