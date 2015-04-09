#ifndef METRICS_HPP
#define METRICS_HPP
#include <vector>
#include <unordered_set>
#include <iostream>
#include "aux.hpp"
#include "types.hpp"

/******************
 *
 * ERROR METRICS
 *
 * ****************/

template<typename Key = unsigned long>
struct RMSE{
    static double eval(const profile_t &predicted, const profile_t &actual){
        double sq_err{.0};
        int n{0};
        for(const auto &act : actual){
            if(predicted.count(act.first) > 0){
                sq_err += std::pow(act.second - predicted.at(act.first), 2);
                ++n;
            }
        }
        return std::sqrt(sq_err / n);
    }
};


/******************
 *
 * RANKING METRICS
 *
 * ****************/

template<std::size_t N = 10, typename Key = unsigned long>
struct Precision{
    static double eval(const std::vector<Key> &ranking, const std::unordered_set<Key> &relevant){
        std::size_t rel_count{0};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        auto length = std::distance(ranking.cbegin(), it_end);
        for(auto it = ranking.cbegin(); it != it_end; ++it){
            if(relevant.count(*it) > 0) ++rel_count;
        }
        return (double) rel_count / length;
    }
};

template<std::size_t N = 10, typename Key = unsigned long>
struct AveragePrecision{
    static double eval(const std::vector<Key> &ranking, const std::unordered_set<Key> &relevant){
        double p_at_k{.0};
        std::size_t rel_count{0}, rank{1};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            if(relevant.count(*it) > 0)
                p_at_k += (double) ++rel_count / rank;
        }
        return p_at_k / relevant.size();
    }
};

template<std::size_t N = 10, typename Key = unsigned long>
struct NDCG{
    static double eval(const std::vector<Key> &ranking, const std::unordered_set<Key> &relevant){
        double dcg{.0}, idcg{.0};
        std::size_t rank{1};
        // Discounted Cumulative Gain
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            if(relevant.count(*it) > 0){
                dcg += 1.0 / std::log2(rank+1);
            }
        }
        std::size_t max_rank = relevant.size() > N ? N : relevant.size();
        // Ideal Discounted Cumulative Gain
        for(rank = 1; rank <= max_rank; ++rank){
            idcg += 1.0 / std::log2(rank+1);
        }
        return dcg / idcg;
    }
};

template<std::size_t N = 10, unsigned HL = 5, typename Key = unsigned long>
struct HLU{
    static double eval(const std::vector<Key> &ranking, const std::unordered_set<Key> &relevant){
        double hlu{.0}, hlu_max{.0};
        std::size_t rank{1};
        // Half Life Utility
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            if(relevant.count(*it) > 0){
                hlu += 1.0 / std::pow(2, (rank-1) / (HL-1));
            }
        }
        std::size_t max_rank = relevant.size() > N ? N : relevant.size();
        // Maximum Half Life Utility
        for(rank = 1; rank <= max_rank; ++rank){
            hlu_max += 1.0 / std::pow(2, (rank-1) / (HL-1));
        }
        return hlu / hlu_max;
    }
};


#endif // METRICS_HPP
