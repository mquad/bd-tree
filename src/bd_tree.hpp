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
#include "../util/basic_log.hpp"

using stats_t = std::map<std::size_t, std::tuple<double, double, int>>;
struct BDIndex{
    using key_t = std::size_t;
    using score_t = std::pair<key_t, double>;
    using entry_t = std::vector<score_t>;
    using bound_t = std::pair<std::size_t, std::size_t>;
    using bound_map_t = std::map<key_t, bound_t>;

    std::map<key_t, entry_t> _index;

    BDIndex():_index{}{}

    // accessors for some basic properties
    std::size_t size() const    {return _index.size();}
    entry_t& at(const key_t &key)              {return _index.at(key);}
    const entry_t& at(const key_t& key) const  {return _index.at(key);}
    entry_t& operator[] (const key_t &key)  {return _index[key];}

    decltype(_index.begin()) begin()    {return _index.begin();}
    decltype(_index.end()) end()    {return _index.end();}
    decltype(_index.cbegin()) cbegin()  {return _index.cbegin();}
    decltype(_index.cend()) cend()  {return _index.cend();}

    // add a new score to the index
    void add_score(const key_t &key, const score_t &score){
        _index[key].push_back(score);
    }

    stats_t compute_all_stats(){
        stats_t stats{};
        for(const auto &entry : _index) update_stats(stats, entry.first);
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
        std::size_t _id;
        // index of the splitter associate with the node wrt to the item_index
        BDIndex::key_t _splitter;
        double _error2;
        unsigned _level;
        std::size_t _num_ratings;
        // pointer to node's parent for hierarchial smoothing
        BDNode * _parent;
        // according to (Golbandi, 2011) descendants of the current node
        // are mapped to the users that either loved, hated or do not known
        // the splitter item. Thus, each node has 3 descendants, corresponding to
        // a partitioning of the set of users associated to it
        std::vector<std::unique_ptr<BDNode>> _children;
        // instead of duplicating the user set for each split, each node keeps only
        // the [left, right) boundaries of the users that are associated to it for each row of the
        // item_index

        // temporary data required during training
        BDIndex::bound_map_t _bounds;
        stats_t _stats;

        BDNode():_id{}, _splitter{}, _error2{}, _num_ratings{}, _parent{nullptr}, _children{}, _bounds{}, _stats{}{}

    };
    using BDNode_ptr = BDNode*;
    using BDNode_cptr = BDNode const*;
    using rating_tuple = std::tuple<std::size_t, std::size_t, int>;
    using group_t = std::vector<BDIndex::key_t>;

    BasicLogger _log;
    BDIndex _item_index;
    BDIndex _user_index;
    std::unique_ptr<BDNode> _root;
    std::size_t _n_users;
    std::size_t _n_items;
    std::size_t _node_counter;
    double _global_mean;
    bool _fix_user_bias;
    double _lambda;
    double _h_smooth;

    BDTree(bool fix_user_bias, double lambda, double h_smooth):
        _log{std::cout}, _item_index{}, _user_index{}, _root{nullptr}, _n_users{}, _n_items{},
        _node_counter{}, _global_mean{}, _fix_user_bias{fix_user_bias}, _lambda{lambda}, _h_smooth{h_smooth}{}


    // builds the item and user indices given the training data
    void init(const std::vector<rating_tuple> &training_data){
        for(const auto &rat : training_data){
            _item_index.add_score(std::get<1>(rat), BDIndex::score_t(std::get<0>(rat), std::get<2>(rat)));
            _user_index.add_score(std::get<0>(rat), BDIndex::score_t(std::get<1>(rat), std::get<2>(rat)));
            _global_mean += std::get<2> (rat);
        }
        _global_mean /= training_data.size();
        _n_items = _item_index.size();
        _n_users = _user_index.size();
        if(_fix_user_bias)  fix_user_biases();
        // initialize the root of the tree
        _root = std::unique_ptr<BDNode>(new BDNode);
        _root->_id = _node_counter++;
        _root->_level = 1u;
        _root->_num_ratings = training_data.size();
        // set root node's bounds
        for(const auto &entry : _item_index)
            _root->_bounds[entry.first] = {0, entry.second.size()};
        std::cout << "ROOT generated" << std::endl;
    }

    void fix_user_biases(){
        for(auto &entry : _user_index){
            double bu{};
            // compute the bias for each user
            for(const auto &score : entry.second)
                bu += score.second;
            bu += _lambda * _global_mean;
            bu /= (entry.second.size() + _lambda);
            // then subtract the bias from each score of the user in the user_idx
            for(auto &score : entry.second)
                score.second -= bu;
        }
    }

    void print_stats(const stats_t &stats){
        for(const auto &s:stats){
            std::cout << "stats item "<< s.first << std::endl;
            std::cout << std::get<0> (s.second) << " " <<
                         std::get<1> (s.second) << " " <<
                         std::get<2> (s.second) << std::endl;
        }
    }

    // computes the squared error given node's statistics
    double error2(const stats_t &stats){
        double err2{};
        for(const auto &s : stats){
            if(std::get<2> (s.second) > 0)
                err2 += std::get<1> (s.second) - std::get<0> (s.second) * std::get<0> (s.second) / std::get<2> (s.second);
        }
        return err2;
    }

    void add_unknown_stats(const stats_t &node_stats, std::vector<stats_t> &group_stats){
        stats_t unknown_stats{};
        // initialize the pointers to the current element for each stats
        std::vector<stats_t::const_iterator> it_stats;
        it_stats.push_back(node_stats.cbegin());
        for(const auto &s : group_stats)
            if(s.size() > 0)    it_stats.push_back(s.cbegin());

        // pass over all the stats simultaneously and update the squared error
        while(it_stats[0] != node_stats.cend()){
            // retrieve current node's stats
            const auto &item = it_stats[0]->first;
            const auto &item_stats = it_stats[0]->second;
            double sum = std::get<0> (item_stats);
            double sum2 = std::get<1> (item_stats);
            double n = std::get<2> (item_stats);
            // subtract groups' stats to get the value for the unknowns
            // note: iterators for groups start from 1 in it_current vector
            for(std::size_t gidx{1}; gidx < it_stats.size(); ++gidx){
                if(it_stats[gidx] != group_stats[gidx].cend()
                        && it_stats[gidx]->first == item){
                    const auto &g_item_stats = it_stats[gidx]->second;
                    sum -= std::get<0> (g_item_stats);
                    sum2 -= std::get<1> (g_item_stats);
                    n -= std::get<2> (g_item_stats);
                    ++it_stats[gidx];
                }
            }
            if(n>0) unknown_stats[item] = stats_t::mapped_type{sum, sum2, n};
            ++it_stats[0];
        }
        // append unknown stats to groups'
        group_stats.push_back(unknown_stats);
    }

    // computes the splitting error for a give candidate item
    // users for each group and respective stats are also returned (unknowns excluded)
    double splitting_error(const BDNode_cptr node,
                           const BDIndex::key_t candidate,
                           std::vector<group_t> &groups,
                           std::vector<stats_t> &group_stats,
                           std::vector<double> &group_errors){
        groups.assign(2, group_t{});
        group_stats.assign(2, stats_t{});

        auto it_left = _item_index.at(candidate).cbegin() + node->_bounds.at(candidate).first;
        auto it_right = _item_index.at(candidate).cbegin() + node->_bounds.at(candidate).second;

        for(auto it_score = it_left; it_score < it_right; ++it_score){
            if(it_score->second >= 4){  // loved item
                groups[0].push_back(it_score->first);
                 _user_index.update_stats(group_stats[0], it_score->first);

            }else{  // hated item
                groups[1].push_back(it_score->first);
                _user_index.update_stats(group_stats[1], it_score->first);
            }
        }
        add_unknown_stats(node->_stats, group_stats);
        //compute the split error
        double split_err2{};
        for(const auto &stats : group_stats){
            double gerr2{error2(stats)};
            group_errors.push_back(gerr2);
            split_err2 += gerr2;
        }
        return split_err2;
    }


    // takes a range of a container of (id, rating) values
    // pre: elements in the range [left, right) must be sorted by "id"
    template<typename It>
    std::vector<BDIndex::bound_t> sort_by_group(It left, It right, const std::vector<group_t> &groups){
        // store the sorted vector chunks in a temp vector (+1 for the unknowns)
        std::vector<std::vector<typename It::value_type>> chunks(groups.size()+1);
        // initialize iterators for each group
        std::vector<group_t::const_iterator> it_groups;
        for(const auto &g : groups)
            it_groups.push_back(g.cbegin());
        // split the input range according to groups
        bool unknown{true};
        for(auto it = left; it != right; ++it, unknown = true){
            for(std::size_t gidx{}; gidx < groups.size(); ++gidx){
                while(it_groups[gidx] < groups[gidx].cend() &&
                      *it_groups[gidx] < it->first)
                    ++it_groups[gidx];
                if(it_groups[gidx] < groups[gidx].cend() &&
                        *it_groups[gidx] == it->first){
                    chunks[gidx].push_back(*it);
                    unknown = false;
                }
            }
            if(unknown) chunks[chunks.size()-1].push_back(*it);
        }
        // recompose the range, now ordered, and generate the bounds wrt the item index
        std::vector<BDIndex::bound_t> bounds;
        std::size_t start{};
        for(const auto &chunk : chunks){
            std::copy(chunk.cbegin(), chunk.cend(), left + start);
            bounds.push_back(BDIndex::bound_t(start, start + chunk.size()));
            start += chunk.size();
        }
        return bounds;
    }

    void split(BDNode_ptr node,
               const std::vector<group_t> &groups,
               const std::vector<stats_t> &group_stats,
               const std::vector<double> &group_errors){
        // fork children, one for each entry in group_stats
        for(std::size_t child_idx{}; child_idx < group_stats.size(); ++child_idx){
            BDNode_ptr child = new BDNode;
            child->_id = _node_counter++;
            child->_level = node->_level + 1;
            child->_parent = node;
            child->_num_ratings = 0u;
            child->_stats = group_stats[child_idx];
            child->_error2 = group_errors[child_idx];
            node->_children.push_back(std::unique_ptr<BDNode>(child));
        }
        // set the boundaries on the item index for each child node
        // according to the group of users they have been assigned to
        for(auto &entry : _item_index){
            auto it_left = entry.second.begin() + node->_bounds.at(entry.first).first;
            auto it_right = entry.second.begin() + node->_bounds.at(entry.first).second;
            const auto g_bounds = sort_by_group(it_left, it_right, groups);
            for(std::size_t gidx{}; gidx < g_bounds.size(); ++gidx){
                node->_children[gidx]->_bounds[entry.first] = g_bounds[gidx];
                node->_children[gidx]->_num_ratings += g_bounds[gidx].second - g_bounds[gidx].first;
            }
        }
    }


    void predict_and_clean(BDNode_ptr node){

    }

    void gdt_r(BDNode_ptr node, const unsigned depth_max, const std::size_t alpha){
        // compute statistics and squared error only for the root node
        // descendant nodes will receive their statistics and squared errors when forked by their parents
        if(node->_level == 1u){
            node->_stats = _user_index.compute_all_stats();
            node->_error2 = error2(node->_stats);
        }
        _log.node(node->_id, node->_level) << "Number of ratings: " << node->_num_ratings << std::endl;
        _log.node(node->_id, node->_level) << "Squared error: " << node->_error2 << std::endl;

        // check termination conditions on node's depth and number of ratings
        if(node->_level == depth_max){
            _log.node(node->_id, node->_level) << "Maximum depth (" << depth_max << ") reached. STOP." << std::endl;
            return;
        }
        if(node->_num_ratings < alpha){
            _log.node(node->_id, node->_level) << "This node has <" << alpha << " ratings. STOP." << std::endl;
            return;
        }

        // lovers and haters groups
        // unknowns is implicitly determined by these two groups and the set of items associate to the node, so we don't store it explicitly
        BDIndex::key_t best_candidate{};
        std::vector<group_t> groups{};
        std::vector<stats_t> group_stats{};
        std::vector<double> group_errors{};
        double min_err{std::numeric_limits<double>::max()};

        // search for the item with the lowest splitting error
        for(BDIndex::key_t candidate{}; candidate < _n_items; ++candidate){
            std::vector<group_t> c_groups{};
            std::vector<stats_t> c_stats{};
            std::vector<double> c_errors{};
            double split_err = splitting_error(node, candidate, c_groups, c_stats, c_errors);
            if(split_err < min_err){
                min_err = split_err;
                best_candidate = candidate;
                groups.swap(c_groups);
                group_stats.swap(c_stats);
                group_errors.swap(c_errors);
            }
        }
        _log.node(node->_id, node->_level) << "Best candidate: " << best_candidate << std::endl;
        _log.node(node->_id, node->_level) << "Splitting error: " << min_err << std::endl;

        // check termination condition on error reduction
        if(min_err >= node->_error2){
            _log.node(node->_id, node->_level) << "The error has not decreased. STOP." << std::endl;
            return;
        }
        node->_splitter = best_candidate;
        // generate children
        split(node, groups, group_stats, group_errors);
        //recursive call
        for(auto &child : node->_children)
            gdt_r(child.get(), depth_max, alpha);
    }

    void build(const unsigned depth_max, const std::size_t alpha){
        // recursively generate the decision tree
        gdt_r(_root.get(), depth_max, alpha);
    }
};


#endif // BDTREE_HPP
