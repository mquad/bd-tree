#ifndef BDTREE_HPP
#define BDTREE_HPP

#include <tuple>
#include <utility>
#include <vector>
#include <limits>
#include <iostream>
#include <memory>

struct BDIndex{
    using key_t = std::size_t;
    using score_t = std::pair<key_t, int>;
    using entry_t = std::vector<score_t>;
    std::vector<entry_t> _index;

    // accessors for some basic properties
    std::size_t size() const    {return _index.size();}
    entry_t operator [](const key_t key)              {return _index[key];}
    const entry_t operator [](const key_t key) const  {return _index[key];}

    // add a new score to the index
    void add_score(const key_t key, const score_t &score){
        while(key + 1 > _index.size())_index.push_back(entry_t());
        _index[key].push_back(score);
    }

    void all_sums(const std::vector<key_t> keys, std::vector<int> &sum, std::vector<int> &sum2, std::vector<int> &counts){
        // compute the sum, squared sum and count of scores for the given keys
        for(const auto user_idx : keys){
            for(auto score : _index[user_idx]){
                sum[score.first] += score.second;
                sum2[score.first] += score.second*score.second;
                counts[score.first]++;
            }
        }
    }

};

struct BDTree{
    struct BDNode{
        // index of the splitter associate with the node wrt to the item_index
        BDIndex::key_t _splitter;
        double _error2;
        unsigned _depth;
        std::size_t _num_ratings;
        std::vector<BDIndex::key_t> _users;
        // pointer to node's parent for hierarchial smoothing
        BDNode * _parent;
        // according to (Golbandi, 2011) descendants of the current node
        // are mapped to the users that either loved, hated or do not known
        // the splitter item. Thus, each node has 3 descendants, corresponding to
        // a partitioning of the set of users associated to it
        BDNode* _children[3];
        // instead of duplicating the user set for each split, each node keeps only
        // the [left, right) boundaries of the users that are associated to it for each row of the
        // item_index
        std::vector<std::pair<BDIndex::key_t, BDIndex::key_t>> _bounds;

    };
    using BDNode_ptr = BDNode*;
    using BDNode_cptr = BDNode const*;

    using rating_tuple = std::tuple<std::size_t, std::size_t, int>;
    BDIndex _item_index;
    BDIndex _user_index;
    BDNode_ptr _root;
    std::size_t _n_users;
    std::size_t _n_items;

    // builds the item and user indices given the training data
    void init(const std::vector<rating_tuple> &training_data){
        for(const auto &rat : training_data){
            _item_index.add_score(std::get<1>(rat), BDIndex::score_t(std::get<0>(rat), std::get<2>(rat)));
            _user_index.add_score(std::get<0>(rat), BDIndex::score_t(std::get<1>(rat), std::get<2>(rat)));
        }
        _n_items = _item_index.size();
        _n_users = _user_index.size();
    }

    double node_error2(const BDNode_cptr node, std::vector<int> &sum, std::vector<int> &sum2, std::vector<int> &counts){
        sum.assign(_n_items, 0);
        sum2.assign(_n_items, 0);
        counts.assign(_n_items, 0);
        _user_index.all_sums(node->_users, sum, sum2, counts);

        // compute the squared error of the node
        double err2{};
        for(std::size_t item_idx{}; item_idx < _n_items; ++item_idx){
            if(counts[item_idx] > 0)
                err2 += static_cast<double>(sum2[item_idx]) -
                        static_cast<double>(sum[item_idx]*sum[item_idx]) / counts[item_idx];
        }
        return err2;
    }

    // computes the squared error for a candidate splitter
    double splitting_error(const BDNode_cptr node,
                           const BDIndex::key_t candidate,
                           const std::vector<int> &sum,
                           const std::vector<int> &sum2,
                           const std::vector<int> &counts,
                           std::vector<std::vector<BDIndex::key_t>> &groups){

        std::vector<int> lsum(_n_items, 0), lsum2(_n_items, 0), lcounts(_n_items, 0);
        std::vector<int> hsum(_n_items, 0), hsum2(_n_items, 0), hcounts(_n_items, 0);

        auto left_bound = _item_index[candidate].cbegin() + node->_bounds[candidate].first;
        auto right_bound = _item_index[candidate].cbegin() + node->_bounds[candidate].second;

        for(auto it_score = left_bound; it_score < right_bound && it_score < _item_index[candidate].cend(); ++it_score){
            if(it_score->second >= 4){ //loved item
                groups[0].push_back(it_score->first);
            }else{ //hated item
                groups[1].push_back(it_score->first);
            }
        }
        // compute lovers and haters rating sums, squared sums and counts
        _user_index.all_sums(groups[0], lsum, lsum2, lcounts);
        _user_index.all_sums(groups[1], hsum, hsum2, hcounts);
        // compute the candidate split squared error
        double err2{};
        for(std::size_t item_idx{}; item_idx < _n_items; item_idx++){
            if(lcounts[item_idx] > 0)   // squared error on lovers
                err2 += static_cast<double>(lsum2[item_idx]) -
                        static_cast<double>(lsum[item_idx] * lsum[item_idx]) / lcounts[item_idx];
            if(hcounts[item_idx] > 0)   // squared error on haters
                err2 += static_cast<double>(hsum2[item_idx]) -
                        static_cast<double>(hsum[item_idx] * hsum[item_idx]) / hcounts[item_idx];
            int ucount = counts[item_idx] - lcounts[item_idx] - hcounts[item_idx];
            if(ucount > 0){ // squared error on unknowns
                int usum = sum[item_idx] - lsum[item_idx] - hsum[item_idx];
                int usum2 = sum2[item_idx] - lsum2[item_idx] - hsum2[item_idx];
                err2 += static_cast<double>(usum2) -
                        static_cast<double>(usum * usum) / ucount;
            }
        }
        return err2;
    }

    void gdt_r(BDNode_ptr root, const unsigned depth_max, const std::size_t alpha){
        // check termination conditions on node's depth and number of ratings
        if(root->_depth + 1 == depth_max || root->_num_ratings < alpha)
            return;

        // compute current node's squared error
        std::vector<int> sum, sum2, counts;
        root->_error2 = node_error2(root, sum, sum2, counts);

        // lovers and haters groups
        // unknowns is implicitly determined by these two groups and the set of items associate to the node, so we don't store it explicitly
        std::vector<BDIndex::key_t> groups[2];
        double min_err = std::numeric_limits<double>::max();

        // search for the item with the lowest splitting error
        for(BDIndex::key_t candidate{}; candidate < _n_items; ++candidate){
            std::vector<std::vector<BDIndex::key_t>> c_groups;
            double split_err = splitting_error(root, candidate, sum, sum2, counts, c_groups);
            if(split_err < min_err){
                root->_splitter = candidate;
                min_err = split_err;
                for(unsigned gidx{}; gidx < 2; ++gidx)
                    groups[gidx].swap(c_groups[gidx]);
            }
        }
        // check termination condition on error reduction
        if(min_err > root->_error2)  return;
        // generate children
    }

    void build(const unsigned depth_max, const std::size_t alpha){
        // initialize the root of the tree
        _root = new BDNode;
        _root->_depth = 0u;
        _root->_parent = nullptr;
        _root->_bounds = std::vector<std::pair<BDIndex::key_t, BDIndex::key_t>>(_item_index.size(), {BDIndex::key_t{0}, std::numeric_limits<BDIndex::key_t>::max()});
        // recursively generate the decision tree
        gdt_r(_root, depth_max, alpha);
    }
};


#endif // BDTREE_HPP
