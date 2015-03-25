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
    size_t n_users{11}, n_items{7};
    auto training_data = make_training_data();
    BDTree bdtree;
    bdtree.init(training_data);

    //check sums and counts
    bdtree._root = new BDTree::BDNode();
    bdtree._root->_users.push_back(1);
    bdtree._root->_users.push_back(7);
    bdtree._root->_users.push_back(9);

    std::vector<int> sum(n_items, 0), sum2(n_items, 0), counts(n_items, 0);
    double node_err = bdtree.node_error2(bdtree._root, sum, sum2, counts);

    ASSERT_EQ(0, sum[0]);
    ASSERT_EQ(7, sum[1]);
    ASSERT_EQ(5, sum[2]);
    ASSERT_EQ(4, sum[3]);
    ASSERT_EQ(0, sum[4]);
    ASSERT_EQ(0, sum[5]);
    ASSERT_EQ(4, sum[6]);

    ASSERT_EQ(0, sum2[0]);
    ASSERT_EQ(25, sum2[1]);
    ASSERT_EQ(13, sum2[2]);
    ASSERT_EQ(6, sum2[3]);
    ASSERT_EQ(0, sum2[4]);
    ASSERT_EQ(0, sum2[5]);
    ASSERT_EQ(16, sum2[6]);

    ASSERT_EQ(0, counts[0]);
    ASSERT_EQ(2, counts[1]);
    ASSERT_EQ(2, counts[2]);
    ASSERT_EQ(3, counts[3]);
    ASSERT_EQ(0, counts[4]);
    ASSERT_EQ(0, counts[5]);
    ASSERT_EQ(1, counts[6]);

    //check node properties
    double eps = 1e-9;
    double true_err = (25.0 - 7.0*7.0/2) +
            (13.0 - 5.0*5.0/2) + (6.0 - 4.0*4.0/3) + (16.0 - 4.0*4.0/1);
    ASSERT_TRUE(std::abs(node_err - true_err) < eps);


}
