#ifndef BDTREE_HPP
#define BDTREE_HPP

#include <cmath>
#include <vector>
#include <map>
#include <limits>
#include <iostream>
#include <memory>
#include <cassert>
#include <utility>
#include "../util/basic_log.hpp"

constexpr bool almost_eq(double lhs, double rhs, double eps = 1e-12) {
    return std::abs(lhs - rhs) < eps;
}

struct stats_t{
    double _sum;
    double _sum2;
    int _n;
    stats_t(double sum, double sum2, int n) : _sum{sum}, _sum2{sum2}, _n{n}{}
    stats_t() : stats_t{.0, .0, 0}{}
};

struct score_t{
    std::size_t _id;
    double _rating;
    score_t(std::size_t id, double rating) : _id{id}, _rating{rating}{}
    score_t() : score_t(0u, .0){}
    friend bool operator ==(const score_t &lhs, const score_t &rhs){
        return (lhs._id == rhs._id) && (almost_eq(lhs._rating, rhs._rating));
    }
    friend bool operator !=(const score_t &lhs, const score_t &rhs){
        return !(lhs == rhs);
    }
};

struct bound_t{
    std::size_t _left, _right;
    bound_t(std::size_t left, std::size_t right) : _left{left}, _right{right}{}
    bound_t() : _left{0u}, _right{0u}{}
    std::size_t size() const {return _right - _left;}
    friend bool operator ==(const bound_t &lhs, const bound_t &rhs){
        return (lhs._left == rhs._left) && (rhs._right == rhs._right);
    }
    friend bool operator !=(const bound_t &lhs, const bound_t &rhs){
        return !(lhs == rhs);
    }
};

using stat_map_t = std::map<std::size_t, stats_t>;
using bound_map_t = std::map<key_t, bound_t>;

struct BDIndex{
    using key_t = std::size_t;
    using entry_t = std::vector<score_t>;

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

    stat_map_t root_stats(){
        stat_map_t stats{};
        for(const auto &entry : _index) update_stats(stats, entry.first);
        return stats;
    }

    // updates the stats with all the values associated to a given key
    void update_stats(stat_map_t &stats, const key_t key){
        for(const auto &score : _index[key]){
            auto &s = stats[score._id];
            s._sum += score._rating;
            s._sum2 += score._rating * score._rating;
            ++s._n;
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
        std::unique_ptr<std::map<std::size_t, double>> _predictions;

        BDNode():
            _id{}, _splitter{}, _error2{}, _num_ratings{}, _parent{nullptr}, _children{},
            _predictions{nullptr}{}


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
            _item_index.add_score(std::get<1>(rat), score_t(std::get<0>(rat), std::get<2>(rat)));
            _user_index.add_score(std::get<0>(rat), score_t(std::get<1>(rat), std::get<2>(rat)));
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

    }

    void fix_user_biases(){
        for(auto &entry : _user_index){
            double bu{};
            // compute the bias for each user
            for(const auto &score : entry.second)
                bu += score._rating;
            bu += _lambda * _global_mean;
            bu /= (entry.second.size() + _lambda);
            // then subtract the bias from each score of the user in the user_idx
            for(auto &score : entry.second)
                score._rating -= bu;
        }
    }

    void print_stats(const stat_map_t &stats){
        for(const auto &s:stats){
            std::cout << "stats item "<< s.first << std::endl;
            std::cout << s.second._sum << " " <<
                         s.second._sum2 << " " <<
                         s.second._n << std::endl;
        }
    }

    // computes the squared error given node's statistics
    double error2(const stat_map_t &stat_map){
        double err2{};
        for(const auto &stats : stat_map){
            const auto &s = stats.second;
            if(s._n > 0)    err2 += s._sum2 - s._sum * s._sum / s._n;
        }
        return err2;
    }

    void add_unknown_stats(const stat_map_t &node_stats, std::vector<stat_map_t> &group_stats){
        stat_map_t unknown_stats{};
        // initialize the pointers to the current element for each stats
        std::vector<stat_map_t::const_iterator> it_stats;
        it_stats.push_back(node_stats.cbegin());
        for(const auto &s : group_stats)
            if(s.size() > 0)    it_stats.push_back(s.cbegin());

        // pass over all the stats simultaneously and update the squared error
        while(it_stats[0] != node_stats.cend()){
            // retrieve current node's stats
            const auto &item = it_stats[0]->first;
            const auto &item_stats = it_stats[0]->second;
            double sum = item_stats._sum;
            double sum2 = item_stats._sum2;
            int n = item_stats._n;
            // subtract groups' stats to get the value for the unknowns
            // note: iterators for groups start from 1 in it_current vector
            for(std::size_t gidx{1}; gidx < it_stats.size(); ++gidx){
                if(it_stats[gidx] != group_stats[gidx].cend()
                        && it_stats[gidx]->first == item){
                    const auto &g_item_stats = it_stats[gidx]->second;
                    sum -= g_item_stats._sum;
                    sum2 -= g_item_stats._sum2;
                    n -= g_item_stats._n;
                    ++it_stats[gidx];
                }
            }
            if(n>0) unknown_stats[item] = stats_t{sum, sum2, n};
            ++it_stats[0];
        }
        // append unknown stats to groups'
        group_stats.push_back(unknown_stats);
    }

    // computes the splitting error for a give candidate item
    // users for each group and respective stats are also returned (unknowns excluded)
    double splitting_error(const BDIndex::key_t candidate,
                           const stat_map_t &parent_stats,
                           const bound_map_t &parent_bounds,
                           std::vector<group_t> &groups,
                           std::vector<stat_map_t> &group_stats,
                           std::vector<double> &group_errors){
        groups.assign(2, group_t{});
        group_stats.assign(2, stat_map_t{});

        auto it_left = _item_index.at(candidate).cbegin() + parent_bounds.at(candidate)._left;
        auto it_right = _item_index.at(candidate).cbegin() + parent_bounds.at(candidate)._right;

        for(auto it_score = it_left; it_score < it_right; ++it_score){
            if(it_score->_rating >= 4){  // loved item
                groups[0].push_back(it_score->_id);
                 _user_index.update_stats(group_stats[0], it_score->_id);

            }else{  // hated item
                groups[1].push_back(it_score->_id);
                _user_index.update_stats(group_stats[1], it_score->_id);
            }
        }
        add_unknown_stats(parent_stats, group_stats);
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
    std::vector<bound_t> sort_by_group(It left, It right, const std::vector<group_t> &groups){
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
                      *it_groups[gidx] < it->_id)
                    ++it_groups[gidx];
                if(it_groups[gidx] < groups[gidx].cend() &&
                        *it_groups[gidx] == it->_id){
                    chunks[gidx].push_back(*it);
                    unknown = false;
                }
            }
            if(unknown) chunks[chunks.size()-1].push_back(*it);
        }
        // recompose the range, now ordered, and generate the bounds wrt the item index
        std::vector<bound_t> bounds;
        std::size_t start{};
        for(const auto &chunk : chunks){
            std::copy(chunk.cbegin(), chunk.cend(), left + start);
            bounds.push_back(bound_t(start, start + chunk.size()));
            start += chunk.size();
        }
        return bounds;
    }

    void split(BDNode_ptr parent_node,
               const bound_map_t &parent_bounds,
               const std::vector<group_t> &groups,
               const std::vector<stat_map_t> &group_stats,
               const std::vector<double> &group_errors,
               const std::size_t depth_max,
               double alpha){
        auto &children = parent_node->_children;
        // fork children, one for each entry in group_stats
        for(std::size_t child_idx{}; child_idx < group_stats.size(); ++child_idx){
            BDNode_ptr child = new BDNode;
            child->_id = _node_counter++;
            child->_level = parent_node->_level + 1;
            child->_parent = parent_node;
            child->_num_ratings = 0u;
            child->_error2 = group_errors[child_idx];
            compute_predictions(child, group_stats[child_idx]);
            children.push_back(std::unique_ptr<BDNode>(child));
        }
        // compute children boundaries
        std::vector<bound_map_t> child_bounds{3};
        for(auto &entry : _item_index){
            auto it_left = entry.second.begin() + parent_bounds.at(entry.first)._left;
            auto it_right = entry.second.begin() + parent_bounds.at(entry.first)._right;
            const auto g_bounds = sort_by_group(it_left, it_right, groups);
            for(std::size_t gidx{}; gidx < g_bounds.size(); ++gidx){
                child_bounds[gidx][entry.first] = g_bounds[gidx];
                children[gidx]->_num_ratings += g_bounds[gidx].size();
            }
        }
        //recursive call
        for(std::size_t child_idx{}; child_idx < children.size(); ++child_idx)
            gdt_r(children[child_idx].get(), group_stats[child_idx], child_bounds[child_idx], depth_max, alpha);
    }

    // cache predictions
    void compute_predictions(BDNode_ptr node, const stat_map_t& stats){
        // cache predictions
        node->_predictions = std::unique_ptr<std::map<std::size_t, double>>(new std::map<std::size_t, double>{});
        if(node->_level > 1){
            for(const auto &p_pred : *(node->_parent->_predictions)){
                double sum{};
                int n{};
                try{
                    const auto &node_stats = stats.at(p_pred.first);
                    sum = node_stats._sum;
                    n = node_stats._n;
                }catch(std::out_of_range &){}
                double pred{(sum + _h_smooth * p_pred.second) / (n + _h_smooth)};
                node->_predictions->insert({p_pred.first, pred});
            }
        }else{
            //root node
            for(const auto &s : stats){
                node->_predictions->insert({s.first, s.second._sum / s.second._n});
            }
        }
    }

    void gdt_r(BDNode_ptr node,
               const stat_map_t &node_stats,
               const bound_map_t &node_bounds,
               const unsigned depth_max,
               const std::size_t alpha){
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
        std::vector<group_t> groups, c_groups;
        std::vector<stat_map_t> group_stats, c_stats;
        std::vector<double> group_errors, c_errors;
        double min_err{std::numeric_limits<double>::max()};

        // search for the item with the lowest splitting error
        for(BDIndex::key_t candidate{}; candidate < _n_items; ++candidate){
            c_groups.clear();
            c_stats.clear();
            c_errors.clear();
            double split_err = splitting_error(candidate, node_stats, node_bounds, c_groups, c_stats, c_errors);
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
        // split the node and perform the recursive call
        split(node, node_bounds, groups, group_stats, group_errors, depth_max, alpha);
    }

    void build(const unsigned depth_max, const std::size_t alpha){
        // compute statistics and squared error only for the root node
        // descendant nodes will receive their statistics and squared errors when forked by their parents
        auto root_stats = _user_index.root_stats();
        _root->_error2 = error2(root_stats);
        // root's bounds
        bound_map_t root_bounds;
        for(const auto &entry : _item_index)
            root_bounds[entry.first] = {0, entry.second.size()};
        // root's predictions
        compute_predictions(_root.get(), root_stats);
        // recursively generate the decision tree
        gdt_r(_root.get(), root_stats, root_bounds, depth_max, alpha);
    }
};


#endif // BDTREE_HPP
