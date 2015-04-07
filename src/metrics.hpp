#ifndef METRICS_HPP
#define METRICS_HPP
#include <vector>
#include <unordered_set>
#include <iostream>
#include "aux.hpp"

/******************
 *
 * ERROR METRICS
 *
 * ****************/

template<typename Key = unsigned long>
struct RMSE{
    static double eval(const std::vector<double> &actual, const std::vector<double> &predicted){
        assert(actual.size() == predicted.size());
        double se{.0};
        for(std::size_t idx{0u}; idx < actual.size(); ++idx)
            se += std::pow(actual[idx] - predicted[idx], 2);
        return std::sqrt(se / actual.size());
    }
};


/******************
 *
 * RANKING METRICS
 *
 * ****************/

template<unsigned N = 10, typename Key = unsigned long>
struct AveragePrecision{
    static double eval(const std::vector<Key> &ranking, const std::unordered_set<Key> &relevant){
        double ap{.0};
        std::size_t rel_count{0}, rank{1};
        auto it_end = ranking.size() < N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            if(relevant.count(*it) > 0)
                ap += (double) ++rel_count / rank;
        }
        return ap / relevant.size();
    }
};

template<usinged N = 10, typename Key = unsigned long>
struct NDCG{
    static double eval(const std::vector<Key> &ranking, const std::unordered_map<Key> &relevant){
        double dcg{.0}, idcg{.0};
        std::size_t rank{1};
        // Discounted Cumulative Gain
        auto it_end = ranking.size() < N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            if(relevant.count(*it) > 0)
                dcg += 1.0 / std::log2(rank+1);
        }
        // Ideal Discounted Cumultaive Gain
        for(rank = 1; rank < relevant.size(); ++rank)
            idcg += 1.0 / std::log2(rank+1);
        return dcg / idcg;
    }
};


#endif // METRICS_HPP
