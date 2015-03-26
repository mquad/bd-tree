#ifndef BDTREE_HPP
#define BDTREE_HPP

#include <tuple>
#include <utility>
#include <vector>
#include <map>
#include <limits>
#include <iostream>
#include <memory>
#include <cassert>

using stats_t = std::map<std::size_t, std::tuple<int, int, int>>;
struct BDIndex{
    using key_t = std::size_t;
    using score_t = std::pair<key_t, int>;
    using entry_t = std::vector<score_t>;
    using bounds_t = std::vector<std::pair<std::size_t, BDIndex::key_t>>;

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

    // returns the sum, squared sum and count of ratings for each entry between the specified bounds
    stats_t compute_stats(const bounds_t &bounds){
        assert(_index.size() == bounds.size());
        stats_t stats;
        for(std::size_t entry_idx{}; entry_idx < _index.size(); ++entry_idx){
            auto it_left = _index[entry_idx].cbegin() + bounds[entry_idx].first;
            auto it_right = _index[entry_idx].cbegin() + bounds[entry_idx].second;
            for(auto it_score = it_left; it_score != it_right; ++it_score){
                auto &s = stats[entry_idx];
                std::get<0> (s) += it_score->second;
                std::get<1> (s) += it_score->second * it_score->second;
                ++std::get<2> (s);
            }
        }
        return stats;
    }

    // updates the stats with all the values associated to a given key
    void update_stats(stats_t &stats, const key_t key){
        for(const auto &score : _index[key]){
            auto &s = stats[score.first];
            std::get<0> (s) += score.second;
            std::get<1> (s) += score.second * score.second;
            ++std::get<2> (s);
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
        stats_t _stats;
        // pointer to node's parent for hierarchial smoothing
        BDNode * _parent;
        // according to (Golbandi, 2011) descendants of the current node
        // are mapped to the users that either loved, hated or do not known
        // the splitter item. Thus, each node has 3 descendants, corresponding to
        // a partitioning of the set of users associated to it
        std::unique_ptr<BDNode> _children[3];
        // instead of duplicating the user set for each split, each node keeps only
        // the [left, right) boundaries of the users that are associated to it for each row of the
        // item_index
        BDIndex::bounds_t _bounds;

    };
    using BDNode_ptr = BDNode*;
    using BDNode_cptr = BDNode const*;
    using rating_tuple = std::tuple<std::size_t, std::size_t, int>;
    using group_t = std::vector<BDIndex::key_t>;

    BDIndex _item_index;
    BDIndex _user_index;
    std::shared_ptr<BDNode> _root;
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
        // initialize the root of the tree
        _root = std::unique_ptr<BDNode>(new BDNode);
        _root->_depth = 0u;
        _root->_num_ratings = training_data.size();
        _root->_parent = nullptr;
        // set root node's bounds
        for(const auto &entry : _item_index._index)
            _root->_bounds.push_back({0, entry.size()});
    }

    // computes the squared error given node's statistics
    double error2(const stats_t &stats){
        double err2{};
        for(const auto &s : stats)
            err2 += static_cast<double>(std::get<1> (s.second)) -
                    static_cast<double>(std::get<0> (s.second) * std::get<0> (s.second)) / std::get<2> (s.second);
        return err2;
    }

    // computes the error for the unknowns given the stats from loved and hated
    double unknown_error2(const stats_t &node_stats, const std::vector<stats_t> &group_stats){
        double err2{};
        // initialize the pointers to the current element for each stats
        std::vector<stats_t::const_iterator> it_current;
        it_current.push_back(node_stats.cbegin());
        for(const auto &s : group_stats)
            it_current.push_back(s.cbegin());
        // pass over all the stats simultaneously and update the squared error
        while(it_current[0] != node_stats.cend()){
            // retrieve current node's stats
            const auto &item = it_current[0]->first;
            const auto &item_stats = it_current[0]->second;
            int sum = std::get<0> (item_stats);
            int sum2 = std::get<1> (item_stats);
            int n = std::get<2> (item_stats);
            // subtract groups' stats to get the value for the unknowns
            // note: iterators for groups start from 1 in it_current vector
            for(std::size_t gidx{1}; gidx <= group_stats.size(); ++gidx){
                if(it_current[gidx]->first == item){
                    const auto &g_item_stats = it_current[gidx]->second;
                    sum -= std::get<0> (g_item_stats);
                    sum2 -= std::get<1> (g_item_stats);
                    n -= std::get<2> (g_item_stats);
                    ++it_current[gidx];
                }
            }
            // update the squared error
            if(n > 0)   err2 += static_cast<double>(sum2) - static_cast<double>(sum * sum) / n;
            ++it_current[0];
        }
        return err2;
    }

    double splitting_error(const BDNode_cptr node,
                           const BDIndex::key_t candidate,
                           std::vector<group_t> &groups,
                           std::vector<stats_t> &group_stats){
        auto it_left = _item_index[candidate].cbegin() + node->_bounds[candidate].first;
        auto it_right = _item_index[candidate].cbegin() + node->_bounds[candidate].second;

        for(auto it_score = it_left; it_score < it_right; ++it_score){
            if(it_score->second >= 4){  // loved item
                groups[0].push_back(it_score->first);
                _user_index.update_stats(group_stats[0], it_score->first);
            }else{  // hated item
                groups[1].push_back(it_score->first);
                _user_index.update_stats(group_stats[1], it_score->first);
            }
        }
        double split_err2{};
        split_err2 += error2(group_stats[0]);   // loved error
        split_err2 += error2(group_stats[1]);   // hated error
        split_err2 += unknown_error2(node->_stats, group_stats);    // unknowns error
        return split_err2;
    }

    void gdt_r(BDNode_ptr node, const unsigned depth_max, const std::size_t alpha){
        // check termination conditions on node's depth and number of ratings
        if(node->_depth + 1 == depth_max || node->_num_ratings < alpha)
            return;
        // compute statistics and squared error only for the root node
        // descendant nodes will receive their statitistics and squared errors from their respective parents
        if(node->_depth == 0){
            node->_stats = _item_index.compute_stats(node->_bounds);
            node->_error2 = error2(node->_stats);
        }

        // lovers and haters groups
        // unknowns is implicitly determined by these two groups and the set of items associate to the node, so we don't store it explicitly
        std::vector<group_t> groups(2);
        std::vector<stats_t> stats(2);
        double min_err = std::numeric_limits<double>::max();

        // search for the item with the lowest splitting error
        for(BDIndex::key_t candidate{}; candidate < _n_items; ++candidate){
            std::vector<group_t> c_groups;
            std::vector<stats_t> c_stats;
            double split_err = splitting_error(node, candidate, c_groups, c_stats);
            if(split_err < min_err){
                node->_splitter = candidate;
                min_err = split_err;
                for(unsigned gidx{}; gidx < 2; ++gidx)
                    groups[gidx].swap(c_groups[gidx]);
            }
        }
        // check termination condition on error reduction
        if(min_err > node->_error2)  return;
        // generate children
    }

    void build(const unsigned depth_max, const std::size_t alpha){
        // recursively generate the decision tree
        gdt_r(_root.get(), depth_max, alpha);
    }
};


#endif // BDTREE_HPP
