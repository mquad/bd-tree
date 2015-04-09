#include <gtest/gtest.h>
#include <cmath>
#include "aux.hpp"
#include "metrics.hpp"

TEST(MetricsTest, APTest){
    EXPECT_TRUE(almost_eq(1, AveragePrecision<4>::eval({3,2,1,4}, {1,2,3,4})));
    EXPECT_TRUE(almost_eq(.25, AveragePrecision<2>::eval({1,2,3,4}, {2,3})));
    EXPECT_TRUE(almost_eq(.0, AveragePrecision<2>::eval({4,1,3,2}, {2,3})));
}

TEST(MetricsTest, NDCGTest){
    EXPECT_TRUE(almost_eq(1, NDCG<4>::eval({3,2,1,4}, {1,2,3,4})));
    EXPECT_TRUE(almost_eq((1/std::log2(2))/(1/std::log2(2) + std::log2(3)),
                          NDCG<2>::eval({1,2,3,4}, {2,3})));
    EXPECT_TRUE(almost_eq(.0, NDCG<2>::eval({4,1,3,2}, {2,3})));
}

TEST(MetricsTest, HLUTest){
    EXPECT_TRUE(almost_eq(1, HLU<4,5>::eval({3,2,1,4}, {1,2,3,4})));
    EXPECT_TRUE(almost_eq((1/std::pow(2,1/4))/(1+1/std::pow(2,1/4)),
                          HLU<2,5>::eval({1,2,3,4}, {2,3})));
    EXPECT_TRUE(almost_eq(.0, HLU<2,5>::eval({4,1,3,2}, {2,3})));
}
