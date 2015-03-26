#include <gtest/gtest.h>
#include <cmath>
#include "bd_tree.hpp"

using rating_t = BDTree::rating_tuple;

std::vector<rating_t> make_training_data(){
    return std::vector<rating_t> {
        rating_t(0, 0, 4),
        rating_t(0, 1, 1),
        rating_t(0, 3, 5),
        rating_t(1, 1, 3),
        rating_t(1, 2, 2),
        rating_t(1, 3, 1),
        rating_t(1, 6, 4),
        rating_t(2, 5, 1),
        rating_t(2, 6, 5),
        rating_t(3, 1, 5),
        rating_t(3, 4, 2),
        rating_t(4, 2, 5),
        rating_t(4, 3, 4),
        rating_t(5, 6, 1),
        rating_t(6, 0, 5),
        rating_t(6, 6, 4),
        rating_t(7, 1, 4),
        rating_t(7, 3, 2),
        rating_t(8, 4, 4),
        rating_t(8, 5, 5),
        rating_t(9, 2, 3),
        rating_t(9, 3, 1),
        rating_t(10, 0, 5)
    };
}

TEST(BDTreeTest, IndexTest){
    size_t n_users{11}, n_items{7};
    auto training_data = make_training_data();
    BDTree bdtree;
    bdtree.init(training_data);
    // check the indices sizes
    ASSERT_EQ(n_items, bdtree._item_index.size());
    ASSERT_EQ(n_users, bdtree._user_index.size());
    //check item index
    ASSERT_EQ(3u, bdtree._item_index[0].size());
    ASSERT_EQ(4u, bdtree._item_index[1].size());
    ASSERT_EQ(3u, bdtree._item_index[2].size());
    ASSERT_EQ(5u, bdtree._item_index[3].size());
    ASSERT_EQ(2u, bdtree._item_index[4].size());
    ASSERT_EQ(2u, bdtree._item_index[5].size());
    ASSERT_EQ(4u, bdtree._item_index[6].size());

    //check some random item index elements
    ASSERT_EQ(BDIndex::score_t(6, 5), bdtree._item_index[0][1]);
    ASSERT_EQ(BDIndex::score_t(7, 4), bdtree._item_index[1][3]);
    ASSERT_EQ(BDIndex::score_t(1, 2), bdtree._item_index[2][0]);
    ASSERT_EQ(BDIndex::score_t(9, 1), bdtree._item_index[3][4]);
    ASSERT_EQ(BDIndex::score_t(3, 2), bdtree._item_index[4][0]);
    ASSERT_EQ(BDIndex::score_t(8, 5), bdtree._item_index[5][1]);
    ASSERT_EQ(BDIndex::score_t(6, 4), bdtree._item_index[6][3]);

    //check user index
    ASSERT_EQ(3u, bdtree._user_index[0].size());
    ASSERT_EQ(4u, bdtree._user_index[1].size());
    ASSERT_EQ(2u, bdtree._user_index[2].size());
    ASSERT_EQ(2u, bdtree._user_index[3].size());
    ASSERT_EQ(2u, bdtree._user_index[4].size());
    ASSERT_EQ(1u, bdtree._user_index[5].size());
    ASSERT_EQ(2u, bdtree._user_index[6].size());
    ASSERT_EQ(2u, bdtree._user_index[7].size());
    ASSERT_EQ(2u, bdtree._user_index[8].size());
    ASSERT_EQ(2u, bdtree._user_index[9].size());
    ASSERT_EQ(1u, bdtree._user_index[10].size());

    //check some random item user elements
    ASSERT_EQ(BDIndex::score_t(3, 5), bdtree._user_index[0][2]);
    ASSERT_EQ(BDIndex::score_t(2, 2), bdtree._user_index[1][1]);
    ASSERT_EQ(BDIndex::score_t(6, 4), bdtree._user_index[1][3]);
    ASSERT_EQ(BDIndex::score_t(4, 2), bdtree._user_index[3][1]);
    ASSERT_EQ(BDIndex::score_t(6, 1), bdtree._user_index[5][0]);
    ASSERT_EQ(BDIndex::score_t(5, 5), bdtree._user_index[8][1]);
    ASSERT_EQ(BDIndex::score_t(0, 5), bdtree._user_index[10][0]);
}

TEST(BDTreeTest, TreeTest){
    auto training_data = make_training_data();
    BDTree bdtree;
    bdtree.init(training_data);

    //check sums and counts
    auto stats = bdtree._item_index.compute_stats(bdtree._root->_bounds);

    EXPECT_EQ(14, std::get<0> (stats[0]));
    EXPECT_EQ(13, std::get<0> (stats[1]));
    EXPECT_EQ(10, std::get<0> (stats[2]));
    EXPECT_EQ(13, std::get<0> (stats[3]));
    EXPECT_EQ(6, std::get<0> (stats[4]));
    EXPECT_EQ(6, std::get<0> (stats[5]));
    EXPECT_EQ(14, std::get<0> (stats[6]));

    EXPECT_EQ(66, std::get<1> (stats[0]));
    EXPECT_EQ(51, std::get<1> (stats[1]));
    EXPECT_EQ(38, std::get<1> (stats[2]));
    EXPECT_EQ(47, std::get<1> (stats[3]));
    EXPECT_EQ(20, std::get<1> (stats[4]));
    EXPECT_EQ(26, std::get<1> (stats[5]));
    EXPECT_EQ(58, std::get<1> (stats[6]));

    EXPECT_EQ(3, std::get<2> (stats[0]));
    EXPECT_EQ(4, std::get<2> (stats[1]));
    EXPECT_EQ(3, std::get<2> (stats[2]));
    EXPECT_EQ(5, std::get<2> (stats[3]));
    EXPECT_EQ(2, std::get<2> (stats[4]));
    EXPECT_EQ(2, std::get<2> (stats[5]));
    EXPECT_EQ(4, std::get<2> (stats[6]));

    //check node properties
    bdtree._root->_stats = stats;
    double node_err = bdtree.error2(stats);
    double eps = 1e-9;
    double true_err = 306 - (196./3 + 169./4 + 100./3 + 169./5 + 36./2 + 36./2 + 196./4);

    EXPECT_TRUE(std::abs(node_err - true_err) < eps);

    //check splitting error
    std::vector<BDTree::group_t> groups(2);
    std::vector<stats_t> g_stats(2);
    double split_err = bdtree.splitting_error(bdtree._root.get(), 1, groups, g_stats);
    double split_err_true = 33.667;

    EXPECT_EQ(split_err_true, split_err);
    EXPECT_TRUE(std::abs(split_err - split_err_true) < 1e-3);

}
