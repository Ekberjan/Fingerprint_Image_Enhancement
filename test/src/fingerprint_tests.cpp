#include "gtest/gtest.h"

using namespace std;

class FingerPrintTest : public ::testing::Test {

protected:
    virtual void SetUp() {
    };

    virtual void TearDown() {
    };

    virtual void verify(int index) {
    }
};

TEST_F(FingerPrintTest, check) {
}
