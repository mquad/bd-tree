#ifndef ABD_TREE_HPP
#define ABD_TREE_HPP
#include <unordered_map>
#include <unordered_set>
#include "aux.hpp"
#include "abd_index.hpp"
#include "d_tree.hpp"
#include "stats.hpp"
#include "types.hpp"

struct ABDNode{
    using stat_map_t = StatMap<id_type, ABDStats>;

    ABDNode(const ABDNode* parent,
            id_type id,
            id_type splitter_id,
            bool is_unknown,
            bool is_leaf,
            unsigned level,
            double quality,
            double split_quality,
            std::size_t num_users,
            std::size_t num_ratings,
            std::size_t top_pop,
            const stat_map_t &stats,
            const std::vector<id_type> &candidates):
        _parent{parent}, _children{},
        _id{id}, _splitter_id{splitter_id}, _is_unknown{is_unknown}, _is_leaf{is_leaf},
        _level{level}, _quality{quality}, _split_quality{split_quality},
        _num_users{num_users}, _num_ratings{num_ratings}, _top_pop{top_pop},
        _stats{std::unique_ptr<stat_map_t>(new stat_map_t{stats})},
        _candidates{std::unique_ptr<std::vector<id_type>>(new std::vector<id_type>(candidates))},
        _users{nullptr}{ }

    ABDNode(id_type id,
            unsigned level,
            std::size_t num_ratings,
            std::size_t num_users,
            std::size_t top_pop,
            const stat_map_t &stats):
        ABDNode(nullptr,
                id,
                -1,
                false,
                true,
                level,
                std::numeric_limits<double>::lowest(),
                std::numeric_limits<double>::lowest(),
                num_users,
                num_ratings,
                top_pop,
                stats,
                std::vector<id_type>{}){}

    ABDNode() : ABDNode(-1, -1, -1, -1, 0, stat_map_t{}){}

    void cache_scores(const double h_smooth);

    std::vector<id_type> candidates() const{
        if(_top_pop > 0){ // Most Popular Sampling
            // sort items by popularity
            std::vector<std::pair<id_type, int>> items_by_pop;
            items_by_pop.reserve(_candidates->size());
            for(const auto &cand : *_candidates)
                if (_stats->count(cand) > 0)
                    items_by_pop.emplace_back(cand, _stats->at(cand)._n);
            std::sort(items_by_pop.begin(),
                      items_by_pop.end(),
                      [](const std::pair<id_type, int> &lhs, const std::pair<id_type, int> &rhs){
                return lhs.second > rhs.second;
            });
            // then peek the first _top_pop ones
            std::vector<id_type> cand;
            cand.reserve(_top_pop);
            auto it_end = _top_pop < items_by_pop.size() ?
                        items_by_pop.cbegin() + _top_pop :
                        items_by_pop.cend();
            for(auto it = items_by_pop.cbegin(); it != it_end; ++it)
                cand.emplace_back(it->first);
            return cand;

        }else{
            return *_candidates;
        }

    }

    bool has_prediction(const id_type &item_id) const{
        if(_predictions == nullptr)
            throw std::runtime_error("Predictions have not been cached. Rebuild the tree with cache_enabled=true.");
        return _predictions->count(item_id) > 0;
    }

    double prediction(const id_type &item_id) const{
        if(_predictions == nullptr)
            throw std::runtime_error("Predictions have not been cached. Rebuild the tree with cache_enabled=true.");
        return _predictions->at(item_id);
    }

    double score(const id_type &item_id) const{
        if(_scores == nullptr)
            throw std::runtime_error("Scores have not been cached. Rebuild the tree with cache_enabled=true.");
        if(_scores->count(item_id) > 0)
            return _scores->at(item_id);
        else
            return std::numeric_limits<double>::lowest();
    }

    bool is_leaf() const{
        return _is_leaf;
    }

    bool has_loved() const{
        return !_children.empty();
    }

    bool has_hated() const{
        return !(_children.size() < 2);
    }

    bool has_unknown() const{
        return !(_children.size() < 3);
    }

    ABDNode* traverse_loved() const{
        if(!has_loved())
            throw std::runtime_error("No LOVED branch from this node.");
        return _children[0].get();
    }

    ABDNode* traverse_hated() const{
        if(!has_hated())
            throw std::runtime_error("No HATED branch from this node.");
        return _children[1].get();
    }

    ABDNode* traverse_unknown() const{
        if(!has_unknown())
            throw std::runtime_error("No UNKNOWN branch from this node.");
        return _children[2].get();
    }

    void free_cache(){
        _stats.reset(nullptr);
        _predictions.reset(nullptr);
        _scores.reset(nullptr);
        _candidates.reset(nullptr);
        _users.reset(nullptr);
        for(auto &child : _children)
            child->free_cache();
    }

    const ABDNode *_parent;
    std::vector<std::unique_ptr<ABDNode>> _children;
    id_type _id;
    id_type _splitter_id;
    bool _is_unknown;
    bool _is_leaf;
    unsigned _level;
    double _quality;
    double _split_quality;
    std::size_t _num_users;
    std::size_t _num_ratings;
    std::size_t _top_pop;
    std::unique_ptr<stat_map_t> _stats;
    std::unique_ptr<std::map<id_type, double>> _predictions;
    std::unique_ptr<std::map<id_type, double>> _scores;
    std::unique_ptr<std::vector<id_type>> _candidates;
    std::unique_ptr<group_t> _users;


};

void ABDNode::cache_scores(const double h_smooth){
    _predictions = std::unique_ptr<std::map<id_type, double> >(new std::map<id_type, double>{});
    _scores = std::unique_ptr<std::map<id_type, double> >(new std::map<id_type, double>{});
    if(_parent != nullptr){
        // user average prediction
        auto it_hint = _predictions->end();
        for(const auto &p_pred : (*_parent->_predictions)){
            if(_stats->count(p_pred.first) > 0){
                it_hint = _predictions->emplace_hint(it_hint, p_pred.first, _stats->at(p_pred.first).pred(p_pred.second, h_smooth));
            }else{
                it_hint = _predictions->emplace_hint(it_hint, p_pred.first, p_pred.second);
            }
        }
        // user unbiased average prediction
        // i.e., average deviation from the user average prediction
        it_hint = _scores->end();
        for(const auto &p_score : (*_parent->_scores)){
            if(_stats->count(p_score.first) > 0){
                it_hint = _scores->emplace_hint(it_hint, p_score.first, _stats->at(p_score.first).score(p_score.second, h_smooth));
            }else{
                it_hint = _scores->emplace_hint(it_hint, p_score.first, p_score.second);
            }
        }
    }else{
        //root node
        auto it_hint = _predictions->end();
        for(const auto &s : (*this->_stats)){
            it_hint = _predictions->emplace_hint(it_hint, s.first, s.second.pred());
        }
        it_hint = _scores->end();
        for(const auto &s : (*this->_stats)){
            it_hint = _scores->emplace_hint(it_hint, s.first, s.second.score());
        }
    }
}

/*
 * Implementation of the Adaptive Bootstrapping Decision Trees method described in
 * "Adaptive Bootstrapping of Recommender Systems Using Decision Trees", Golbandi et. al, WSDM'11
*/

class ABDTree : public DTree<ABDNode>{
protected:
    using index_t = ABDIndex<id_type, ABDStats>;
    using bound_t = typename index_t::bound_t;
    using bound_map_t = hash_map_t<id_type, bound_t>;
    using stat_map_t = typename DTree<ABDNode>::stat_map_t;
public:
    using typename DTree<ABDNode>::node_ptr_t;
    using typename DTree<ABDNode>::node_cptr_t;
public:
    ABDTree(const double bu_reg = 7,
            const double h_smooth = 100,
            const unsigned depth_max = 6,
            const std::size_t ratings_min = 200000,
            const std::size_t top_pop = 0,
            const unsigned num_threads = 1,
            const bool randomize = false,
            const double rand_coeff = 10,
            const bool cache_enabled = true,
            const BasicLogger &log = BasicLogger{std::cout}):
        DTree<ABDNode>(depth_max, ratings_min, num_threads, randomize, rand_coeff, log),
        _item_index{nullptr}, _user_index{nullptr}, _node_bounds{nullptr},
        _bu_reg{bu_reg}, _h_smooth{h_smooth}, _top_pop{top_pop}, _cache_enabled{cache_enabled}, _node_counter{0u}{}

    ~ABDTree(){
        std::cout << "~ABDTree()" << std::endl;
    }

    using DTree<ABDNode>::gdt_r;

    void build() override;
    void build(const std::vector<id_type> &candidates);
    void init(const std::vector<Rating> &training_data, const std::vector<Rating> &validation_data) override;
    void init(const std::vector<Rating> &training_data) override;
    bool traverse(node_ptr_t &node, const profile_t &answers) const override;

    profile_t predict(const node_cptr_t node,
                      const std::vector<id_type> &items) const override{
        profile_t pred;
        pred.set_empty_key(-1);
        for(const auto &item_id : items){
            if(node->has_prediction(item_id))
                pred.insert(std::make_pair(item_id, node->prediction(item_id)));
        }
        return pred;
    }

    std::vector<id_type> ranking(const node_cptr_t node,
                              const std::vector<id_type> &items) const override{
        std::vector<id_type> rank(items);
        std::sort(rank.begin(), rank.end(),
                  [&](const id_type &lhs, const id_type &rhs){
            return node->score(lhs) > node->score(rhs);
        });
        return rank;
    }

    void release_temp() override{
        _root->free_cache();
    }

protected:
    void compute_biases(const double global_mean);
    void compute_root_quality() override;

    template<typename It>
    std::vector<bound_t> sort_by_group(It left,
                                       It right,
                                       std::size_t start,
                                       const std::vector<group_t> &groups);
    void split(node_ptr_t parent,
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
    double squared_error(const stat_map_t &stats) const;
    void unknown_stats(const node_cptr_t node,
                       std::vector<stat_map_t> &group_stats) const;
protected:
    std::unique_ptr<index_t> _item_index;
    std::unique_ptr<index_t> _user_index;
    std::unique_ptr<hash_map_t<id_type, bound_map_t>> _node_bounds;
    double _bu_reg;
    double _h_smooth;
    std::size_t _top_pop;
    bool _cache_enabled;
    unsigned _node_counter;
};

void ABDTree::init(const std::vector<Rating> &training_data, __attribute__((unused)) const std::vector<Rating> &validation_data){
    //NOTE: This method simply calls the standard initialization
    //Validation was not considered in the original work of Golbandi et al..
    init(training_data);
}

void ABDTree::init(const std::vector<Rating> &training_data){
    double global_mean{0};
    _item_index = std::unique_ptr<index_t>(new index_t{});
    _user_index = std::unique_ptr<index_t>(new index_t{});
    for(const auto &rat : training_data){
        _item_index->insert(rat._item_id, ScoreUnbiased{rat._user_id, rat._value, rat._value});
        _user_index->insert(rat._user_id, ScoreUnbiased{rat._item_id, rat._value, rat._value});
        global_mean += rat._value;
    }
    _item_index->sort_all();
    _user_index->sort_all();
    global_mean /= training_data.size();
    compute_biases(global_mean);
    this->_log.log() << "TRAINING:" << std::endl
                 << "Num. users: " << _user_index->size() << std::endl
                 << "Num. items: " << _item_index->size() << std::endl;
    //initialize the root of the tree
    this->_root = std::unique_ptr<ABDNode>(new ABDNode(_node_counter++,
                                                       1,
                                                       training_data.size(),
                                                       _user_index->size(),
                                                       _top_pop,
                                                       _user_index->all_stats()));
    compute_root_quality();
}

void ABDTree::compute_biases(const double global_mean){
    for(auto &entry : *_user_index){
        double bu{};
        // compute the bias for each user
        for(const auto &score : entry.second)
            bu += score._rating;
        bu += _bu_reg * global_mean;
        bu /= (entry.second.size() + _bu_reg);
        // the update the unbiased scores
        for(auto &score : entry.second)
            score._rating_unbiased -= bu;
    }
}

void ABDTree::compute_root_quality(){
    _root->_quality = -squared_error(*_root->_stats);
}

double ABDTree::squared_error(const stat_map_t &stats) const{
    double sq{.0};
    for(const auto &entry : stats)
        sq += entry.second.squared_error();
    return sq;

}

void ABDTree::build(){
    build(_item_index->keys());
}

void ABDTree::build(const std::vector<id_type> &candidates){
    // compute the intersection between candidates and item_index keys
    std::vector<id_type> candidates_sorted(candidates);
    std::sort(candidates_sorted.begin(), candidates_sorted.end());
    std::vector<id_type> item_index_keys = _item_index->keys();
    std::vector<id_type> intersection(std::min(item_index_keys.size(), candidates_sorted.size()));
    auto it = std::set_intersection(candidates_sorted.cbegin(), candidates_sorted.cend(),
                                    item_index_keys.cbegin(), item_index_keys.cend(),
                                    intersection.begin());
    intersection.resize(it - intersection.begin());

    // assign candidates to the root node
    this->_root->_candidates = std::unique_ptr<std::vector<id_type>>(new std::vector<id_type>(intersection));

    // compute root node's bounds
    _node_bounds = std::unique_ptr<hash_map_t<id_type, bound_map_t>>(new hash_map_t<id_type, bound_map_t>{});
    _node_bounds->set_empty_key(-1);
    for(const auto &entry : *_item_index){
        if(_node_bounds->count(this->_root->_id) == 0)
            (*_node_bounds)[this->_root->_id].set_empty_key(-1);
        (*_node_bounds)[this->_root->_id].insert(std::make_pair(entry.first, typename index_t::bound_t(0, entry.second.size())));
    }

    // cache root's scores
    if(_cache_enabled)  this->_root->cache_scores(_h_smooth);
    // generate the decision tree
    this->_log.node(this->_root->_id, this->_root->_level)
            << "Num.users: " << this->_root->_num_users
            << "\tNum.ratings: " << this->_root->_num_ratings
            << "\tQuality: " << this->_root->_quality << std::endl;

    gdt_r(this->_root.get());
    //free memory allocated for temporary indices
    _item_index.reset(nullptr);
    _user_index.reset(nullptr);
    _node_bounds.reset(nullptr);

}

void ABDTree::split(node_ptr_t parent,
                    const id_type splitter_id,
                    const double splitter_quality,
                    std::vector<group_t> &groups,
                    std::vector<double> &g_qualities,
                    std::vector<stat_map_t> &g_stats){
    // update parent node
    parent->_splitter_id = splitter_id;
    parent->_split_quality = splitter_quality;
    parent->_is_leaf = false;

    // fork children, one for each entry in group_stats
    auto &children = parent->_children;
    std::size_t u_num_users = parent->_num_users;
    for(std::size_t child_idx{}; child_idx < g_stats.size(); ++child_idx){
        node_ptr_t child = new ABDNode;
        child->_parent = parent;
        child->_id = _node_counter++;
        child->_candidates = std::unique_ptr<std::vector<id_type>>(new std::vector<id_type>(*parent->_candidates));
        child->_level = parent->_level + 1;
        child->_is_leaf = true;
        child->_num_ratings = 0u;
        child->_quality = g_qualities[child_idx];
        child->_top_pop = parent->_top_pop;
        // store child stats temporarely
        child->_stats = std::unique_ptr<stat_map_t>(new stat_map_t{g_stats[child_idx]});
        if(_cache_enabled)  child->cache_scores(_h_smooth);
        if(child_idx < groups.size()){
            child->_num_users = groups[child_idx].size();
            u_num_users -= child->_num_users;
        }else{
            child->_num_users = u_num_users;
            child->_is_unknown = true;
        }
        children.push_back(std::unique_ptr<ABDNode>(child));
    }
    // compute children boundaries
    for(auto &entry : *_item_index){
        const auto &item_bounds = (*_node_bounds)[parent->_id][entry.first];
        auto it_left = entry.second.begin() + item_bounds._left;
        auto it_right = entry.second.begin() + item_bounds._right;
        const auto g_bounds = sort_by_group(it_left, it_right, item_bounds._left, groups);
        for(std::size_t gidx{}; gidx < g_bounds.size(); ++gidx){
            if(_node_bounds->count(children[gidx]->_id) == 0)
                    (*_node_bounds)[children[gidx]->_id].set_empty_key(-1);
            (*_node_bounds)[children[gidx]->_id][entry.first] = g_bounds[gidx];
            children[gidx]->_num_ratings += g_bounds[gidx].size();
        }
    }
}

double ABDTree::split_quality(const node_cptr_t node,
                              const id_type splitter_id,
                              std::vector<group_t> &groups,
                              std::vector<double> &g_qualities,
                              std::vector<stat_map_t> &g_stats) const {
    groups.clear();
    g_qualities.clear();
    g_stats.clear();
    groups.assign(2, group_t{});
    g_stats.assign(2, stat_map_t());

    auto it_left = _item_index->at(splitter_id).cbegin() + (*_node_bounds)[node->_id][splitter_id]._left;
    auto it_right = _item_index->at(splitter_id).cbegin() + (*_node_bounds)[node->_id][splitter_id]._right;

    for(auto it_score = it_left; it_score < it_right; ++it_score){
        if(it_score->_rating >= 4){  // loved item
            groups[0].push_back(it_score->_id);
             _user_index->update_stats(g_stats[0], it_score->_id);

        }else{  // hated item
            groups[1].push_back(it_score->_id);
            _user_index->update_stats(g_stats[1], it_score->_id);
        }
    }
    unknown_stats(node, g_stats);
    //compute the split error on the training data
    double split_quality{.0};
    for(const auto &stats : g_stats){
        double gq{-squared_error(stats)};
        g_qualities.push_back(gq);
        split_quality += gq;
    }
    return split_quality;

}

bool ABDTree::traverse(node_ptr_t &node,
                       const profile_t &answers) const {
    auto &answers_non_const = const_cast<profile_t&>(answers);
    bool to_unknown = false;
    if(!node->is_leaf()){ // while not at leaf node
        if(answers.count(node->_splitter_id) == 0){// unknown item
            node = node->_children.back().get();
            to_unknown = true;
        }else{
            double &rating = answers_non_const[node->_splitter_id];
            if(rating >= 4)
                node = node->_children[0].get(); // loved
            else
                node = node->_children[1].get(); // hated
        }
    }else{
        node = nullptr;
    }
    return !to_unknown;
}

// takes a range of a container of (id, rating) values
// pre: elements in the range [left, right) must be sorted by "id" ascending
// pre: vectors in groups must be sorted in ascending order
template<typename It>
std::vector<ABDTree::bound_t> ABDTree::sort_by_group(It left,
                                                     It right,
                                                     std::size_t start,
                                                     const std::vector<group_t> &groups){
    assert(is_ordered(left, right));
    // store the sorted vector chunks in a temp vector (+1 for the unknowns)
    std::vector<std::vector<typename It::value_type>> chunks;
    chunks.reserve(groups.size()+1);
    for(std::size_t gidx{0}; gidx < groups.size()+1; ++gidx){
        chunks.push_back(std::vector<typename It::value_type>{});
        chunks.back().reserve(std::distance(left, right));  // reserve more memory than actually needed..
    }
    // initialize iterators for each group
    std::vector<group_t::const_iterator> it_groups;
    it_groups.reserve(groups.size());
    for(const auto &g : groups)
        it_groups.push_back(g.cbegin());
    // split the input range according to groups
    bool unknown{true};
    for(auto it = left; it != right; ++it, unknown = true){
        for(std::size_t gidx{0}; gidx < groups.size(); ++gidx){
            while(it_groups[gidx] < groups[gidx].cend() &&
                  *it_groups[gidx] < it->_id)
                ++it_groups[gidx];
            if(it_groups[gidx] < groups[gidx].cend() &&
                    *it_groups[gidx] == it->_id){
                chunks[gidx].push_back(*it);
                unknown = false;
            }
        }
        if(unknown) chunks.back().push_back(*it);
    }
    // recompose the range, now ordered, and generate the bounds wrt the item index
    std::vector<bound_t> bounds;
    bounds.reserve(chunks.size());
    auto it_start = left;
    for(const auto &chunk : chunks){
        std::copy(chunk.cbegin(), chunk.cend(), it_start);
        bounds.push_back(bound_t(start, start + chunk.size()));
        assert(is_ordered(it_start, it_start + chunk.size()));
        it_start += chunk.size();
        start += chunk.size();
    }
    return bounds;
}

void ABDTree::unknown_stats(const node_cptr_t node,
                            std::vector<stat_map_t> &group_stats) const{
    group_stats.push_back(ABDNode::stat_map_t());
    // initialize the pointers to the current element for each stats
    std::vector<decltype(node->_stats->cbegin())> it_stats;
    it_stats.reserve(group_stats.size());
    // const_iterator for emplace_hint on the stat_map for the unknown statistics
    auto it_hint = group_stats.back().end();

    for(const auto &s : group_stats)
        it_stats.push_back(s.cbegin());
    // pass over all the stats simultaneously and
    // compute the stats for the unknown branch of the tree
    for(const auto &parent_stats : (*node->_stats)){
        // retrieve current node's stats
        const auto &item = parent_stats.first;
        auto unknown_stats = parent_stats.second;
        // subtract groups' stats to get the value for the unknowns
        for(std::size_t gidx{0}; gidx < it_stats.size(); ++gidx){
            if(it_stats[gidx] != group_stats[gidx].cend()
                    && it_stats[gidx]->first == item){
                unknown_stats -= it_stats[gidx]->second;
                ++it_stats[gidx];
            }
        }
        if(unknown_stats._n > 0)
            it_hint = group_stats.back().emplace_hint(it_hint, item, unknown_stats);
    }
#ifdef DEBUG
    for(const auto &entry : node->_stats){
        assert(entry.second._sum ==
               group_stats[0].at(entry.first)._sum +
               group_stats[1].at(entry.first)._sum +
               group_stats[0].at(entry.first)._sum);
        assert(entry.second._sum2 ==
               group_stats[0].at(entry.first)._sum2 +
               group_stats[1].at(entry.first)._sum2 +
               group_stats[0].at(entry.first)._sum2);
        assert(entry.second._n ==
               group_stats[0].at(entry.first)._n +
               group_stats[1].at(entry.first)._n +
               group_stats[0].at(entry.first)._n);
    }
#endif
}



#endif // ABD_TREE_HPP
