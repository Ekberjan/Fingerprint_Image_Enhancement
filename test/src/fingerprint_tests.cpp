#include "gtest/gtest.h"
#include "common.h"
#include "fpenhancement.h"

using namespace std;

class FingerPrintTest : public ::testing::Test {

protected:
    virtual void SetUp() {
    };

    virtual void TearDown() {
    };

};

class ExtractFingerPrintsTest : public FingerPrintTest {};

class PostProcessingFilteringTest : public FingerPrintTest {};


TEST_F(ExtractFingerPrintsTest, emptyInputMatrixError) {
    cv::Mat empty;
    FPEnhancement fpEnhancement;

    EXPECT_THROW({
                     try
                     {
                         fpEnhancement.extractFingerPrints(empty);
                     }
                     catch( const std::invalid_argument& e )
                     {
                         // and this tests that it has the correct message
                         EXPECT_STREQ( "The input matrix should not be empty.", e.what() );
                         throw;
                     }
                 }, std::invalid_argument );
}

TEST_F(PostProcessingFilteringTest, emptyInputMatrixError) {
    cv::Mat empty;
    FPEnhancement fpEnhancement;

    EXPECT_THROW({
                     try
                     {
                         fpEnhancement.postProcessingFilter(empty);
                     }
                     catch( const std::invalid_argument& e )
                     {
                         // and this tests that it has the correct message
                         EXPECT_STREQ( "The input matrix should not be empty.", e.what() );
                         throw;
                     }
                 }, std::invalid_argument );
}


