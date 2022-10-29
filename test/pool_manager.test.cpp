#include <gtest/gtest.h>

#include <string>
#include <iostream>

#include <alf/pool_manager.hpp>

using namespace alf;

TEST(PoolHandleTest, CreatingObjectPtr) {
	std::string path = "pathString";
	auto * poolHandler = new Pool_manager(path);
	EXPECT_TRUE(poolHandler != nullptr);
	delete poolHandler;
}

TEST(PoolHandleTest, LoadingPoolArff) {
	std::string path = TEST_RESOURCE_DIR "/example_dataset.arff";
	auto poolHandler = Pool_manager(path);
	poolHandler.load();
	auto matrix = poolHandler.getMatrix();
	EXPECT_TRUE(matrix->n_rows == 4);
	EXPECT_TRUE(matrix->n_cols == 6);
	EXPECT_TRUE(matrix->n_elem == 24);
}

TEST(PoolHandleTest, LoadingPoolWithWrongPath) {
	std::string path = "non_exist_example_tls_pool.csv";
	auto poolHandler = Pool_manager(path);
	EXPECT_THROW(poolHandler.load(), std::runtime_error);
}

TEST(PoolHandleTest, RemoveSampleFromMatrix) {
	std::string path = TEST_RESOURCE_DIR "/example_dataset.arff";
	auto poolHandler = Pool_manager(path);
	poolHandler.load();
	auto matrix = poolHandler.getMatrix();
	EXPECT_TRUE(matrix->n_rows == 4);
	EXPECT_TRUE(matrix->n_cols == 6);
	EXPECT_TRUE(matrix->n_elem == 24);
	poolHandler.removeSample({0});
	EXPECT_TRUE(matrix->n_rows == 4);
	EXPECT_TRUE(matrix->n_cols == 5);
}

TEST(PoolHandleTest, RemoveSampleFromMatrixOutOfBounds) {
	std::string path = TEST_RESOURCE_DIR "/example_dataset.arff";
	auto poolHandler = Pool_manager(path);
	poolHandler.load();
	auto matrix = poolHandler.getMatrix();
	EXPECT_TRUE(matrix->n_rows == 4);
	EXPECT_TRUE(matrix->n_elem == 24);
	EXPECT_TRUE(matrix->n_cols == 6);
	EXPECT_THROW(poolHandler.removeSample({10000}), std::out_of_range);
}

TEST(PoolHandleTest, RemoveVectorOfSamples) {
	std::string path = TEST_RESOURCE_DIR "/example_dataset.arff";
	auto poolHandler = Pool_manager(path);
	poolHandler.load();
	auto matrix = poolHandler.getMatrix();
	EXPECT_TRUE(matrix->n_rows == 4);
	EXPECT_TRUE(matrix->n_elem == 24);
	EXPECT_TRUE(matrix->n_cols == 6);
	poolHandler.removeSample({0, 1, 2});
	EXPECT_TRUE(matrix->n_rows == 4);
	EXPECT_TRUE(matrix->n_cols == 3);
}