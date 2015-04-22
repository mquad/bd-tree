#ifndef D_TREE_HPP
#define D_TREE_HPP
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <omp.h>
#include "ratings.hpp"
#include "stats.hpp"
#include "types.hpp"
#include "stopwatch.hpp"
#include "../util/basic_log.hpp"
template<typename N>
class DTree{
public:
    using node_ptr_t = N*;
    using node_cptr_t = N const*;
    using id_t = typename N::id_t;
    using stat_map_t = typename N::stat_map_t;

public:
    DTree(const unsigned depth_max,
          const unsigned ratings_min,
          const unsigned num_threads = 1,
          const bool randomize = false,
          const double rand_coeff = 10,
          const BasicLogger &log = BasicLogger{std::cout}):
        _root{nullptr}, _mt{nullptr},
        _depth_max{depth_max}, _ratings_min{ratings_min}, _num_threads{num_threads},
        _randomize{randomize}, _rand_coeff{rand_coeff},
        _log{log}{}



    virtual ~DTree(){
        std::cout << "~DTree()" << std::endl;
    }
    virtual void build() = 0;
    virtual void init(const std::vector<Rating> &training_data) = 0;
    virtual void init(const std::vector<Rating> &training_data, const std::vector<Rating> &validation_data) = 0;
    void gdt_r(node_ptr_t node);

    virtual profile_t predict(const node_cptr_t node,
                              const std::vector<id_t> &items) const = 0;

    virtual std::vector<id_t> ranking(const node_cptr_t node,
                                      const std::vector<id_t> &items) const = 0;

    virtual node_ptr_t traverse(const node_ptr_t node, const profile_t &answers) const = 0;

    node_ptr_t root()           {return _root.get();}
    node_ptr_t root() const     {return _root.get();}
    unsigned depth_max() const  {return _depth_max;}

protected:
    virtual void compute_root_quality(node_ptr_t node) = 0;

    void find_splitter(const node_cptr_t node,
                       id_t &splitter,
                       double &quality,
                       std::vector<group_t> &groups,
                       std::vector<double> &g_qualities,
                       std::vector<stat_map_t> &g_stats);
    virtual void split(node_ptr_t node,
                       const id_t splitter_id,
                       const double splitter_quality,
                       std::vector<group_t> &groups,
                       std::vector<double> &g_qualities,
                       std::vector<stat_map_t> &g_stats) = 0;
    virtual double split_quality(const node_cptr_t node,
                                 const id_t splitter_id,
                                 std::vector<group_t> &groups,
                                 std::vector<double> &g_qualities,
                                 std::vector<stat_map_t> &g_stats) const = 0;
    void rnd_init(){
        std::random_device rd;
        _mt = std::unique_ptr<std::mt19937>(new std::mt19937(rd()));
    }
protected:
    std::unique_ptr<N> _root;
    std::unique_ptr<std::mt19937> _mt;
    unsigned _depth_max;
    std::size_t _ratings_min;
    unsigned _num_threads;
    bool _randomize;
    double _rand_coeff;
    BasicLogger _log;

};

template<typename N>
void DTree<N>::gdt_r(node_ptr_t node){
    // check termination conditions
    if(node->_level >= _depth_max){
        _log.node(node->_id, node->_level) << "Maximum depth (" << _depth_max << ") reached. STOP." << std::endl;
        return;
    }
    if(node->_num_ratings < _ratings_min){
        _log.node(node->_id, node->_level) << "This node has < " << _ratings_min << " ratings. STOP." << std::endl;
        return;
    }
    if(node->_num_users < 2) {
        _log.node(node->_id, node->_level) << "This node has < 2 users. No split can be found. STOP." << std::endl;
        return;
    }

    id_t splitter{};
    double quality{};
    std::vector<group_t> groups{};
    std::vector<double> g_qualities{};
    std::vector<stat_map_t> g_stats{};

    stopwatch sw;
    sw.reset();sw.start();
    find_splitter(node, splitter, quality, groups, g_qualities, g_stats);
    sw.stop();
    _log.node(node->_id, node->_level) << "Splitter found in " << sw.elapsed_ms() / 1000.0 << " sec."
                                       << "\tId: " << splitter
                                       << "\tQuality: " << quality << std::endl;

    if(quality <= node->_quality){
        _log.node(node->_id, node->_level) << "The quality after split will not increase. STOP." << std::endl;
        return;
    }
    split(node, splitter, quality, groups, g_qualities, g_stats);
    for(const auto &child : node->_children){
        this->_log.node(child->_id, child->_level)
                << "Num.users: " << child->_num_users
                << "\tNum.ratings: " << child->_num_ratings
                << "\tQuality: " << child->_quality << std::endl;
        //recursive call
        gdt_r(child.get());
    }

}

template<typename N>
void DTree<N>::find_splitter(const node_cptr_t node,
                             id_t &splitter,
                             double &quality,
                             std::vector<group_t> &groups,
                             std::vector<double> &g_qualities,
                             std::vector<stat_map_t> &g_stats){
    const auto &candidates = node->candidates();
    if(candidates.empty()) return;

    if(!_randomize){    // pick the best quality candidate
        std::vector<std::vector<group_t>> cand_groups{_num_threads};
        std::vector<std::vector<double>> cand_g_qualities{_num_threads};
        std::vector<std::vector<stat_map_t>> cand_g_stats{_num_threads};
        std::vector<std::pair<id_t, double>> cand_best_qualities{_num_threads, std::make_pair(id_t{},
                                                                                              std::numeric_limits<double>::lowest())};

        // compute the qualiy of each candidate in parallel
    #pragma omp parallel num_threads(_num_threads)
        {
            std::vector<group_t> c_groups;
            std::vector<double> c_qualities;
            std::vector<stat_map_t> c_stats;
    #pragma omp single
            {
                for(auto it_cand = candidates.cbegin(); it_cand != candidates.cend(); ++it_cand){
    #pragma omp task firstprivate(it_cand)
                    {
                        unsigned thread_id = omp_get_thread_num();
                        double cand_quality = split_quality(node,
                                                            *it_cand,
                                                            c_groups,
                                                            c_qualities,
                                                            c_stats);
                        if(cand_quality > cand_best_qualities[thread_id].second){
                            cand_best_qualities[thread_id].first = *it_cand;
                            cand_best_qualities[thread_id].second = cand_quality;
                            cand_groups[thread_id].swap(c_groups);
                            cand_g_qualities[thread_id].swap(c_qualities);
                            cand_g_stats[thread_id].swap(c_stats);
                        }
                    }
                }
            }
        }

        auto it_best = cand_best_qualities.cbegin();
        for(auto it = cand_best_qualities.cbegin()+1; it != cand_best_qualities.cend(); ++it){
            if(it->second > it_best->second)  it_best = it;
        }

        const auto best_thread = std::distance(cand_best_qualities.cbegin(), it_best);
        splitter = it_best->first;
        quality = it_best->second;
        groups.swap(cand_groups[best_thread]);
        g_qualities.swap(cand_g_qualities[best_thread]);
        g_stats.swap(cand_g_stats[best_thread]);
    }else{
        // pick a candidate with probability proportianal to his enhancement in quality
        //to reduce the memory footprint, we store just the candidate qualities, then recompute the groups just for the chosen one
        std::vector<std::pair<id_t, double>> cand_qualities{candidates.size(), std::make_pair(id_t{},
                                                                                              std::numeric_limits<double>::lowest())};

        // compute the qualiy of each candidate in parallel
    #pragma omp parallel num_threads(_num_threads)
        {
            std::vector<group_t> c_groups;
            std::vector<double> c_qualities;
            std::vector<stat_map_t> c_stats;
    #pragma omp single
            {
                for(auto it_cand = candidates.cbegin(); it_cand != candidates.cend(); ++it_cand){
    #pragma omp task firstprivate(it_cand)
                    {
                        const auto cand_idx = std::distance(candidates.cbegin(), it_cand);
                        cand_qualities[cand_idx] =
                                std::make_pair(*it_cand,
                                               split_quality(node,
                                                             *it_cand,
                                                             c_groups,
                                                             c_qualities,
                                                             c_stats));
                    }
                }
            }
        }

        std::vector<double> cum_probs;
        cum_probs.reserve(candidates.size());
        // compute the probabilies for each node
        for(auto it_qual = cand_qualities.cbegin(); it_qual != cand_qualities.cend(); ++it_qual){
            double prob = std::pow(std::max(.0, it_qual->second - node->_quality), _rand_coeff);
            if(std::distance(cand_qualities.cbegin(), it_qual) > 0)
                cum_probs.push_back(cum_probs.back() + prob);
            else
                cum_probs.push_back(prob);
        }
        //pick an element at random
        if(_mt == nullptr) rnd_init();
        std::uniform_real_distribution<double> rnd(0, cum_probs.back());
        double p = rnd(*_mt);
        auto it_prob = cum_probs.cbegin();
        auto prob_end = cum_probs.cend();
        while(it_prob != prob_end-1 &&
              *it_prob <= p) ++it_prob;
        const auto &chosen = cand_qualities.at(std::distance(cum_probs.cbegin(), it_prob));
        splitter = chosen.first;
        quality = chosen.second;

        // recompute the groups, qualities and stats for the chosen splitter
        std::vector<group_t> c_groups;
        std::vector<double> c_qualities;
        std::vector<stat_map_t> c_stats;

        split_quality(node,splitter,c_groups,c_qualities,c_stats);

        groups.swap(c_groups);
        g_qualities.swap(c_qualities);
        g_stats.swap(c_stats);
    }
}

#endif // D_TREE_HPP
