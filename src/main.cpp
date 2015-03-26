#include <iostream>
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

int main()
{
    auto training_data = make_training_data();
    BDTree bdtree;
    bdtree.init(training_data);
    bdtree.build(2, 0);

}

