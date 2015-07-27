#include <gtest/gtest.h>
#include <cmath>
#include "aux.hpp"
#include "metrics.hpp"

TEST(MetricsTest, APTest){
    EXPECT_TRUE(almost_eq(1.0, AveragePrecision<4>::eval({3,2,1,4}, {{1,5},{2,5},{3,5},{4,5}})));
    EXPECT_TRUE(almost_eq(.25, AveragePrecision<2>::eval({1,2,3,4}, {{2,5},{3,5}})));
    EXPECT_TRUE(almost_eq(.0, AveragePrecision<2>::eval({4,1,3,2}, {{2,5},{3,5}})));
}

TEST(MetricsTest, NDCGTest){
    EXPECT_TRUE(almost_eq(1, NDCG<4>::eval({3,2,1,4}, {{1,5},{2,5},{3,5},{4,5}})));
    EXPECT_TRUE(almost_eq(((std::pow(2,5)-1)/std::log2(3))/((std::pow(2,5)-1)/std::log2(2) + (std::pow(2,5)-1)/std::log2(3)),
                          NDCG<2>::eval({1,2,3,4}, {{2,5},{3,5}})));
    EXPECT_TRUE(almost_eq(.0, NDCG<2>::eval({4,1,3,2}, {{2,5},{3,5}})));
}

TEST(MetricsTest, HLUTest){
    EXPECT_TRUE(almost_eq(1.0, HLU<4,5>::eval({3,2,1,4}, {{1,5},{2,5},{3,5},{4,5}})));
    EXPECT_TRUE(almost_eq((5/std::pow(2,1/4))/(5+5/std::pow(2,1/4)),
                          HLU<2,5>::eval({1,2,3,4}, {{2,5},{3,5}})));
    EXPECT_TRUE(almost_eq(.0, HLU<2,5>::eval({4,1,3,2}, {{2,5},{3,5}})));
}
