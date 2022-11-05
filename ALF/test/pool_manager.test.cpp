#include <gtest/gtest.h>

#include <string>
#include <iostream>

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
