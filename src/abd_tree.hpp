#ifndef ABD_TREE_HPP
#define ABD_TREE_HPP
#include <unordered_map>
#include "aux.hpp"
#include "abd_index.hpp"
#include "d_tree.hpp"
#include "stats.hpp"

template<typename S>
struct ABDNode{
    using id_t = std::size_t;
    using stat_map_t = StatMap<id_t, S>;
    using pred_map_t = std::map<id_t, double>;

    ABDNode(const std::vector<ABDNode<S>*> &children,
            id_t id,
            id_t splitter_id,
            unsigned level,
            double quality,
            double split_quality,
            std::size_t num_users,
            std::size_t num_ratings,
            std::size_t top_pop,
            const stat_map_t &stats,
            const pred_map_t &predictions):
        _children{}, _id{id}, _splitter_id{splitter_id}, _level{level}, _quality{quality}, _split_quality{split_quality},
        _num_users{num_users}, _num_ratings{num_ratings},_top_pop{top_pop},
        _stats{stats}, _predictions{predictions}{
        for(auto node : children)
            _children.push_back(std::unique_ptr<ABDNode<S>>(node));
    }

    ABDNode(id_t id,
            unsigned level,
            std::size_t num_ratings,
            std::size_t num_users,
            std::size_t top_pop,
            const stat_map_t &stats):
        ABDNode(std::vector<ABDNode<S>*>{},
                id,
                -1,
                level,
                std::numeric_limits<double>::lowest(),
                std::numeric_limits<double>::lowest(),
                num_users,
                num_ratings,
                top_pop,
                stats,
                pred_map_t{}){
        //compute node quality with the given stats
        this->_quality = compute_quality(this->_stats);
    }

    ABDNode() : ABDNode(-1, -1, -1, -1, 0, stat_map_t{}){}

    void cache_predictions(const ABDNode * const parent, const double h_smooth);

    std::vector<id_t> candidates() const{
        if(_top_pop > 0){ // Most Popular Sampling
            // sort items by popularity
            std::vector<std::pair<id_t, int>> items_by_pop;
            items_by_pop.reserve(_stats.size());
            for(const auto &entry : _stats) items_by_pop.emplace_back(entry.first, entry.second._n);
            std::sort(items_by_pop.begin(),
                      items_by_pop.end(),
                      [](const std::pair<id_t, int> &lhs, const std::pair<id_t, int> &rhs){
                return lhs.second > rhs.second;
            });
            // then peek the first _top_pop ones
            std::vector<id_t> cand;
            cand.reserve(_top_pop);
            auto it_end = _top_pop < items_by_pop.size() ?
                        items_by_pop.cbegin() + _top_pop :
                        items_by_pop.cend();
            for(auto it = items_by_pop.cbegin(); it != it_end; ++it)
                cand.emplace_back(it->first);
            return cand;

        }else{
            std::vector<id_t> cand;
            cand.reserve(this->_stats.size());
            auto it_end = this->_stats.cend();
            for(auto it = this->_stats.cbegin(); it != it_end; ++it)
                cand.emplace_back(it->first);
            return cand;
        }

    }

    double prediction(const id_t item_id) const{
        return _predictions.at(item_id);
    }


    std::vector<std::unique_ptr<ABDNode<S>>> _children;
    id_t _id;
    id_t _splitter_id;
    unsigned _level;
    double _quality;
    double _split_quality;
    std::size_t _num_users;
    std::size_t _num_ratings;
    std::size_t _top_pop;
    stat_map_t _stats;
    pred_map_t _predictions;

};
template<typename S>
void ABDNode<S>::cache_predictions(const ABDNode * const parent, const double h_smooth){
    if(this->_level > 1){
        for(const auto &p_pred : parent->_predictions){
            double sum{};
            int n{};
            try{
                const auto &node_stats = this->_stats.at(p_pred.first);
                sum = node_stats._sum;
                n = node_stats._n;
            }catch(std::out_of_range &){}
            double pred{(sum + h_smooth * p_pred.second) / (n + h_smooth)};
            _predictions.emplace(p_pred.first, pred);
        }
    }else{
        //root node
        for(const auto &s : this->_stats){
            _predictions.emplace(s.first, s.second._sum / s.second._n);
        }
    }
}


template<typename N>
class ABDTree : public DTree<N>{
protected:
    using index_t = ABDIndex<typename N::id_t, ABDStats>;
    using bound_t = typename index_t::bound_t;
    using bound_map_t = std::unordered_map<typename index_t::key_t, bound_t>;
    using stat_map_t = typename DTree<N>::stat_map_t;
    using typename DTree<N>::node_ptr_t;
    using typename DTree<N>::node_cptr_t;
    using typename DTree<N>::id_t;
public:
    ABDTree(const std::size_t min_ratings = 200000,
            const double bu_reg = 7,
            const double h_smooth = 100,
            const unsigned depth_max = 6,
            const std::size_t top_pop = 0,
            const unsigned num_threads = 1,
            const bool randomize = false,
            const double rand_coeff = 10,
            const BasicLogger &log = BasicLogger{std::cout}):
        DTree<N>(depth_max, num_threads, randomize, rand_coeff, log),
        _item_index{}, _user_index{}, _node_bounds{},
        _min_ratings{min_ratings}, _bu_reg{bu_reg}, _h_smooth{h_smooth}, _top_pop{top_pop}, _node_counter{0u}{}

    ~ABDTree(){}

    using DTree<N>::init;
    using DTree<N>::gdt_r;

    void build() override;
    void init(const std::vector<rating_t> &training_data) override;
    void split(node_ptr_t node,
               const id_t splitter_id,
               const double splitter_quality,
               std::vector<group_t> &groups,
               std::vector<double> &g_qualities,
               std::vector<stat_map_t> &g_stats) override;
    double split_quality(const node_cptr_t node,
                         const id_t splitter_id,
                         std::vector<group_t> &groups,
                         std::vector<double> &g_qualities,
                         std::vector<stat_map_t> &g_stats) const override;
    node_cptr_t traverse(const profile_t &answers) const override;
    double predict(const node_cptr_t node, const id_t item_id) const{
        return node->prediction(item_id);
    }


protected:
    void compute_biases(const double global_mean);
    template<typename It>
    std::vector<bound_t> sort_by_group(It left,
                                       It right,
                                       std::size_t start,
                                       const std::vector<group_t> &groups);
    void unknown_stats(const node_cptr_t node,
                       std::vector<stat_map_t> &group_stats) const;
protected:
    index_t _item_index;
    index_t _user_index;
    std::map<id_t, bound_map_t> _node_bounds;
    std::size_t _min_ratings;
    double _bu_reg;
    double _h_smooth;
    std::size_t _top_pop;
    unsigned _node_counter;
};

template<typename N>
void ABDTree<N>::init(const std::vector<rating_t> &training_data){
    double global_mean{0};
    for(const auto &rat : training_data){
        _item_index.insert(rat._item_id, ScoreUnbiased{rat._user_id, rat._value, rat._value});
        _user_index.insert(rat._user_id, ScoreUnbiased{rat._item_id, rat._value, rat._value});
        global_mean += rat._value;
    }
    global_mean /= training_data.size();
    _item_index.sort_all();
    _user_index.sort_all();
    compute_biases(global_mean);
    this->_log.log() << "TRAINING:" << std::endl
                 << "Num. users: " << _user_index.size() << std::endl
                 << "Num. items: " << _item_index.size() << std::endl;
    //initialize the root of the tree
    this->_root = std::unique_ptr<N>(new N(_node_counter++,
                                           1,
                                           training_data.size(),
                                           _user_index.size(),
                                           _top_pop,
                                           _user_index.all_stats()));
}

template<typename N>
void ABDTree<N>::compute_biases(const double global_mean){
    for(auto &entry : _user_index){
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

template<typename N>
void ABDTree<N>::build(){
    // compute root node's bounds
    for(const auto &entry : _item_index)
        _node_bounds[this->_root->_id].emplace(entry.first, typename index_t::bound_t(0, entry.second.size()));
    // cache root's predictions
    this->_root->cache_predictions(nullptr, _h_smooth);
    // generate the decision tree
    this->_log.node(this->_root->_id, this->_root->_level)
            << "Num.users: " << this->_root->_num_users
            << "\tNum.ratings: " << this->_root->_num_ratings
            << "\tQuality: " << this->_root->_quality << std::endl;

    gdt_r(this->_root.get());
}

template<typename N>
void ABDTree<N>::split(node_ptr_t node,
                       const id_t splitter_id,
                       const double splitter_quality,
                       std::vector<group_t> &groups,
                       std::vector<double> &g_qualities,
                       std::vector<stat_map_t> &g_stats){
    node->_splitter_id = splitter_id;
    node->_split_quality = splitter_quality;
    auto &children = node->_children;
    // fork children, one for each entry in group_stats
    std::size_t u_num_users = node->_num_users;
    for(std::size_t child_idx{}; child_idx < g_stats.size(); ++child_idx){
        node_ptr_t child = new N;
        child->_id = _node_counter++;
        child->_level = node->_level + 1;
        child->_num_ratings = 0u;
        child->_quality = g_qualities[child_idx];
        child->_stats.swap(g_stats[child_idx]);
        child->_top_pop = node->_top_pop;
        child->cache_predictions(node, _h_smooth);
        if(child_idx < groups.size()){
            child->_num_users = groups[child_idx].size();
            u_num_users -= child->_num_users;
        }
        children.push_back(std::unique_ptr<N>(child));
    }
    children.back()->_num_users = u_num_users;
    // compute children boundaries
    for(auto &entry : _item_index){
        const auto &item_bounds = _node_bounds.at(node->_id).at(entry.first);
        auto it_left = entry.second.begin() + item_bounds._left;
        auto it_right = entry.second.begin() + item_bounds._right;
        const auto g_bounds = sort_by_group(it_left, it_right, item_bounds._left, groups);
        for(std::size_t gidx{}; gidx < g_bounds.size(); ++gidx){
            _node_bounds[children[gidx]->_id][entry.first] = g_bounds[gidx];
            children[gidx]->_num_ratings += g_bounds[gidx].size();
        }
    }

    //recursive call
    for(auto &child : children){
        this->_log.node(child->_id, child->_level)
                << "Num.users: " << child->_num_users
                << "\tNum.ratings: " << child->_num_ratings
                << "\tQuality: " << child->_quality << std::endl;
        gdt_r(child.get());
    }
}

template<typename N>
double ABDTree<N>::split_quality(const node_cptr_t node,
                                 const id_t splitter_id,
                                 std::vector<group_t> &groups,
                                 std::vector<double> &g_qualities,
                                 std::vector<stat_map_t> &g_stats) const {
    groups.clear();
    g_qualities.clear();
    g_stats.clear();
    groups.assign(2, group_t{});
    g_stats.assign(2, stat_map_t());

    auto it_left = _item_index.at(splitter_id).cbegin() + _node_bounds.at(node->_id).at(splitter_id)._left;
    auto it_right = _item_index.at(splitter_id).cbegin() + _node_bounds.at(node->_id).at(splitter_id)._right;

    for(auto it_score = it_left; it_score < it_right; ++it_score){
        if(it_score->_rating >= 4){  // loved item
            groups[0].push_back(it_score->_id);
             _user_index.update_stats(g_stats[0], it_score->_id);

        }else{  // hated item
            groups[1].push_back(it_score->_id);
            _user_index.update_stats(g_stats[1], it_score->_id);
        }
    }
    unknown_stats(node, g_stats);
    //compute the split error
    double split_quality{.0};
    for(const auto &stats : g_stats){
        double gq{compute_quality(stats)};
        g_qualities.push_back(gq);
        split_quality += gq;
    }
    return split_quality;

}
template<typename N>
typename ABDTree<N>::node_cptr_t ABDTree<N>::traverse(const profile_t &answers) const {
    node_cptr_t node_ptr = this->_root.get();
    while(!node_ptr->_children.empty()){ // while not at leaf node
        if(answers.count(node_ptr->_splitter_id) == 0){
            // unknown item
            node_ptr = node_ptr->_children.back().get();
        }else{
            double rating = answers.at(node_ptr->_splitter_id);
            if(rating >= 4) node_ptr = node_ptr->_children[0].get(); // loved
            else node_ptr = node_ptr->_children[1].get(); // hated
        }
    }
    return node_ptr;
}

// takes a range of a container of (id, rating) values
// pre: elements in the range [left, right) must be sorted by "id" asc
// pre: vectors in groups must be sorted in asc ordere
template<typename N>
template<typename It>
std::vector<typename ABDTree<N>::bound_t> ABDTree<N>::sort_by_group(It left,
                                                                 It right,
                                                                 std::size_t start,
                                                                 const std::vector<group_t> &groups){
    assert(is_ordered(left, right));
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


template<typename N>
void ABDTree<N>::unknown_stats(const node_cptr_t node,
                               std::vector<stat_map_t> &group_stats) const{
    group_stats.push_back(typename N::stat_map_t());
    // initialize the pointers to the current element for each stats
    std::vector<decltype(node->_stats.cbegin())> it_stats;
    it_stats.reserve(group_stats.size());
    for(const auto &s : group_stats)
        it_stats.push_back(s.cbegin());
    // pass over all the stats simultaneously and
    // compute the stats for the unknown branch of the treee
    for(const auto &parent_stats : node->_stats){
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
            group_stats.back().emplace(item, unknown_stats);
    }
}



#endif // ABD_TREE_HPP
