#ifndef RANKTREE_HPP
#define RANKTREE_HPP
#include <algorithm>
#include <unordered_set>
#include "abd_tree.hpp"
#include "aux.hpp"

template<typename Key, typename Metric, int RelTh = 4>
struct RankingIndex{
    using entry_t = std::unordered_set<Key>;
protected:
    std::map<Key, entry_t> _index;
public:
    // accessors for some basic properties
    std::size_t size() const    {return _index.size();}
    entry_t& at(const key_t &key)              {return _index.at(key);}
    const entry_t& at(const key_t& key) const  {return _index.at(key);}
    entry_t& operator[] (const key_t &key)     {return _index[key];}

    decltype(_index.begin()) begin()            {return _index.begin();}
    decltype(_index.end()) end()                {return _index.end();}
    decltype(_index.cbegin()) cbegin() const    {return _index.cbegin();}
    decltype(_index.cend()) cend() const        {return _index.cend();}

    void insert(const Key &key, const Key &item, const double &rating){
        if(rating >= RelTh)
            _index[key].insert(item);
    }

    //pre: keys in each group in groups are sorted in descending order
    void evaluate(const std::vector<std::vector<Key>> &ranking, const std::vector<group_t> &groups, std::vector<double> &g_qualities) const{
        assert(ranking.size() == groups.size()+1); // the unknown users are not stored into groups directly to save memory
        assert(is_ordered(groups[0].cbegin(), groups[0].cend()));
        assert(is_ordered(groups[1].cbegin(), groups[1].cend()));
        g_qualities.assign(groups.size()+1, .0);
        std::vector<group_t::const_iterator> g_it;
        g_it.reserve(groups.size());
        for(const auto &g : groups) g_it.emplace_back(g.cbegin());
        for(const auto &entry : _index){
            bool unknown{true};
            for(std::size_t gidx{0u}; gidx < g_it.size(); ++gidx){
                while(g_it[gidx] < groups[gidx].cend() &&
                      *g_it[gidx] <= entry.first)  ++g_it[gidx];
                if(g_it[gidx] < groups[gidx].cend() &&
                        *g_it[gidx] == entry.first){
                    unknown = false;
                    g_qualities[gidx] += Metric::eval(ranking[gidx], entry.second);
                }
            }
            if(unknown) g_qualities.back() += Metric::eval(ranking.back(), entry.second);
        }
    }
};

template<typename Node, typename R>
class RankTree : public ABDTree<Node>{
protected:
    using typename ABDTree<Node>::id_t;
    using typename ABDTree<Node>::index_t;
    using typename ABDTree<Node>::node_ptr_t;
    using typename ABDTree<Node>::node_cptr_t;
    using typename ABDTree<Node>::stat_map_t;

public:
    // inherith the constructor
    using ABDTree<Node>::ABDTree;
    ~RankTree(){}

    void init(const std::vector<Rating> &training_data) override{
        // initialize the validation index with the validation data
        for(const auto &rat : training_data)
            _ranking_index.insert(rat._user_id, rat._item_id, rat._value);
        ABDTree<Node>::init(training_data);
    }

    void compute_root_quality(node_ptr_t node) override;

    double split_quality(const node_cptr_t node,
                         const id_t splitter_id,
                         std::vector<group_t> &groups,
                         std::vector<double> &g_qualities,
                         std::vector<stat_map_t> &g_stats) const override;
protected:
    using ABDTree<Node>::unknown_stats;
    using ABDTree<Node>::_item_index;
    using ABDTree<Node>::_user_index;
    R _ranking_index;

};



template<typename N, typename R>
void RankTree<N, R>::compute_root_quality(node_ptr_t node){
    double q{.0};
    std::vector<id_t> ranks;
    build_ranking(node->_stats, ranks);
    for(const auto &entry : _ranking_index)
        q += _ranking_index.evaluate(ranks, entry.first);
    node->_quality = q;
}

template<typename N, typename R>
double RankTree<N, R>::split_quality(const node_cptr_t node,
                                     const id_t splitter_id,
                                     std::vector<group_t> &groups,
                                     std::vector<double> &g_qualities,
                                     std::vector<stat_map_t> &g_stats) const{
    groups.clear();
    g_qualities.clear();
    g_stats.clear();
    groups.assign(2, group_t{});
    g_stats.assign(2, stat_map_t());

    auto it_left = this->_item_index.at(splitter_id).cbegin() + this->_node_bounds.at(node->_id).at(splitter_id)._left;
    auto it_right = this->_item_index.at(splitter_id).cbegin() + this->_node_bounds.at(node->_id).at(splitter_id)._right;

    for(auto it_score = it_left; it_score < it_right; ++it_score){
        if(it_score->_rating >= 4){  // loved item
            groups[0].push_back(it_score->_id);
            this->_user_index.update_stats(g_stats[0], it_score->_id);

        }else{  // hated item
            groups[1].push_back(it_score->_id);
            this->_user_index.update_stats(g_stats[1], it_score->_id);
        }
    }
    unknown_stats(node, g_stats);
    std::vector<std::vector<id_t>> ranks{g_stats.size()};
    if(node->_level > 1){
        for(std::size_t gidx{0u}; gidx < g_stats.size(); ++gidx)
            build_ranking(g_stats[gidx], node->_scores, this->_h_smooth, ranks[gidx]);
    }else{
        for(std::size_t gidx{0u}; gidx < g_stats.size(); ++gidx)
            build_ranking(g_stats[gidx], ranks[gidx]);
    }
    _ranking_index.evaluate(ranks, groups, g_qualities);
    return std::accumulate(g_qualities.cbegin(), g_qualities.cend(), .0, std::plus<double>());
}

#endif // RANKTREE_HPP
