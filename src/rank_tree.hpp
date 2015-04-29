#ifndef RANKTREE_HPP
#define RANKTREE_HPP
#include <algorithm>
#include <cstdio>
#include "abd_tree.hpp"
#include "aux.hpp"
#include "stopwatch.hpp"
#include "types.hpp"

template<typename Key, typename Metric>
struct RankIndex{
    using ranking_t = std::vector<Key>;
    using relevance_t = hash_map_t<Key, double>;
protected:
    hash_map_t<Key, relevance_t> _relevances;
    hash_map_t<Key, ranking_t> _best_rankings;
public:
    RankIndex(){
        _relevances.set_empty_key(-1);
        _best_rankings.set_empty_key(-1);
    }

    void insert(const Key &key, const Key &item, const double &rating){
        if(_relevances.count(key) == 0){
            _relevances[key].set_empty_key(-1);
        }
        _relevances[key].insert(std::make_pair(item, rating));
    }

    // return the -unordered- key vector
    std::vector<Key> keys(){
        return extract_keys(_relevances);
    }

    double evaluate_all(const std::vector<Key> &global_ranking){
        const auto &users = extract_keys(_relevances);
        // persist the optimal ranking for user to speed up further evaluations
        for(const auto &user : users)
            _best_rankings[user] = keys_sorted_by_relevance(_relevances[user]);
        return evaluate_users(global_ranking, keys());
    }

    double evaluate_users(const std::vector<Key> &global_ranking, const std::vector<Key> &users){
        double m{.0};
        for(const auto &u : users)
            m += evaluate_user(global_ranking, u);
        return m;
    }

    double evaluate_user(const std::vector<Key> &global_ranking, const Key &user){
        auto &user_relevance = _relevances[user];
        auto &best_ranking = _best_rankings[user];
        // compute the ranking only on items rated by the user
        std::vector<Key> user_ranking;
        user_ranking.reserve(user_relevance.size());
        for(const auto key : global_ranking)
            if(user_relevance.count(key) > 0)
                user_ranking.push_back(key);
        return Metric::eval_fast(user_ranking, best_ranking, user_relevance);
    }
protected:
    std::vector<Key> keys_sorted_by_relevance(const relevance_t &relevance){
        auto &relevance_non_const = const_cast<relevance_t&>(relevance);
        std::vector<Key> best_ranking(extract_keys(relevance));
        std::sort(best_ranking.begin(), best_ranking.end(),
                  [&](const Key &lhs, const Key &rhs){
            return relevance_non_const[lhs] > relevance_non_const[rhs];
        });
        return best_ranking;
    }
};

template<typename R>
class RankTree : public ABDTree{
protected:
    using ABDTree::index_t;
    using ABDTree::stat_map_t;
public:
    using ABDTree::node_ptr_t;
    using ABDTree::node_cptr_t;
public:
    // inherith the constructor
    using ABDTree::ABDTree;
    ~RankTree(){
        std::cout << "~RankTree()" << std::endl;
    }

    void init(const std::vector<Rating> &training_data, const std::vector<Rating> &validation_data){
        //TODO: caching is forced for the time. Support for optional caching to be added in future.
        _cache_enabled = true;
        // initialize the validation index with the validation data
        _ranking_index = std::unique_ptr<R>(new R{});
        for(const auto &rat : validation_data){
            _ranking_index->insert(rat._user_id, rat._item_id, rat._value);
        }
        ABDTree::init(training_data);
        // initialize root _users member
        this->_root->_users = std::unique_ptr<group_t>(new group_t{});
        this->_root->_users->reserve(_user_index->size());
        for(const auto &entry : *_user_index)
            this->_root->_users->push_back(entry.first);

    }

    void init(const std::vector<Rating> &training_data) override{
        //TODO: caching is forced for the time. Support for optional caching to be added in future.
        _cache_enabled = true;
        _ranking_index = std::unique_ptr<R>(new R{});
        for(const auto &rat : training_data){
            _ranking_index->insert(rat._user_id, rat._item_id, rat._value);
        }
        ABDTree::init(training_data);
        // initialize root _users member
        this->_root->_users = std::unique_ptr<group_t>(new group_t{});
        this->_root->_users->reserve(_user_index->size());
        for(const auto &entry : *_user_index)
            this->_root->_users->push_back(entry.first);
    }

    void build(){
        ABDTree::build();
        //free ranking index's memory
        _ranking_index.reset(nullptr);
    }

    void build(const std::vector<id_type> &candidates){
        ABDTree::build(candidates);
        //free ranking index's memory
        _ranking_index.reset(nullptr);
    }

protected:
    void compute_root_quality() override;

    void split(node_ptr_t node,
               const id_type splitter_id,
               const double splitter_quality,
               std::vector<group_t> &groups,
               std::vector<double> &g_qualities,
               std::vector<stat_map_t> &g_stats) override;

    double split_quality(const node_cptr_t node,
                         const id_type splitter_id,
                         std::vector<group_t> &groups,
                         std::vector<double> &g_qualities,
                         std::vector<stat_map_t> &g_stats) const override;
    void unknown_users(const node_cptr_t node, std::vector<group_t> &groups) const;
protected:
    using ABDTree::unknown_stats;
    using ABDTree::_item_index;
    using ABDTree::_user_index;
    std::unique_ptr<R> _ranking_index;
};

template<typename R>
void RankTree<R>::compute_root_quality(){
    _root->_quality = _ranking_index->evaluate_all(rank_all_items(*_root->_stats));
}

template<typename R>
void RankTree<R>::split(node_ptr_t node,
                             const id_type splitter_id,
                             const double splitter_quality,
                             std::vector<group_t> &groups,
                             std::vector<double> &g_qualities,
                             std::vector<stat_map_t> &g_stats){

    // groups now contains users also for the unknown branch
    // need to remove it for the base split method to work properly
    std::vector<group_t> base_groups(groups.size()-1);
    std::copy(groups.cbegin(), groups.cend()-1, base_groups.begin());

    ABDTree::split(node, splitter_id, splitter_quality, base_groups, g_qualities, g_stats);
    // explicitly save the ids of the users of each children node
    for(std::size_t gidx{0u}; gidx < groups.size(); ++gidx)
        node->_children[gidx]->_users = std::unique_ptr<group_t>(new group_t(groups[gidx]));

    assert(node->_children[0]->_users->size() == node->_children[0]->_num_users);
    assert(node->_children[1]->_users->size() == node->_children[1]->_num_users);
    assert(node->_children[2]->_users->size() == node->_children[2]->_num_users);
}

template<typename R>
double RankTree<R>::split_quality(const node_cptr_t node,
                                     const id_type splitter_id,
                                     std::vector<group_t> &groups,
                                     std::vector<double> &g_qualities,
                                     std::vector<stat_map_t> &g_stats) const{
    stopwatch sw;
    sw.reset(); sw.start();

    groups.clear();
    g_qualities.clear();
    g_stats.clear();
    groups.assign(2, group_t{});
    g_stats.assign(2, stat_map_t());

    auto it_left = this->_item_index->at(splitter_id).cbegin() + (*_node_bounds)[node->_id][splitter_id]._left;
    auto it_right = this->_item_index->at(splitter_id).cbegin() + (*_node_bounds)[node->_id][splitter_id]._right;

    for(auto it_score = it_left; it_score < it_right; ++it_score){
        if(it_score->_rating >= 4){  // loved item
            groups[0].push_back(it_score->_id);
            this->_user_index->update_stats(g_stats[0], it_score->_id);
        }else{  // hated item
            groups[1].push_back(it_score->_id);
            this->_user_index->update_stats(g_stats[1], it_score->_id);
        }
    }
    unknown_stats(node, g_stats);
    unknown_users(node, groups);
    double quality{.0};
    for(std::size_t gidx{0u}; gidx < groups.size(); ++gidx){
        if(node->_level > 1){
            g_qualities.emplace_back(
                        _ranking_index->evaluate_users(
                            rank_all_items(g_stats[gidx], *node->_scores, this->_h_smooth),
                            groups[gidx]));
        }else{
            g_qualities.emplace_back(
                        _ranking_index->evaluate_users(
                            rank_all_items(g_stats[gidx]),
                            groups[gidx]));
        }
        quality += g_qualities.back();
    }
    return quality;
}

template<typename R>
void RankTree<R>::unknown_users(const node_cptr_t node,
                                      std::vector<group_t> &groups) const{
    groups.push_back(*node->_users); //init with parent's users
    auto &unknown_users = groups.back();
    assert(is_ordered(unknown_users.begin(), unknown_users.end()));
    for(std::size_t gidx{0}; gidx < groups.size()-1; ++gidx){
        assert(is_ordered(groups[gidx].begin(), groups[gidx].end()));
        group_t diff_result(unknown_users.size());
        auto it = std::set_difference(unknown_users.begin(), unknown_users.end(),
                                      groups[gidx].begin(), groups[gidx].end(),
                                      diff_result.begin());
        diff_result.resize(std::distance(diff_result.begin(), it));
        unknown_users.swap(diff_result);
    }
    assert(unknown_users.size() == node->_num_users - groups[0].size() - groups[1].size());
}
#endif // RANKTREE_HPP
