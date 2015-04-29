#ifndef METRICS_HPP
#define METRICS_HPP
#include <algorithm>
#include <vector>
#include <iostream>
#include "aux.hpp"
#include "types.hpp"

/******************
 *
 * ERROR METRICS
 *
 * ***************
*/


template<typename Key = id_type>
struct RMSE{
    static double eval(profile_t &predicted, profile_t &actual){
        double sq_err{.0};
        int n{0};
        for(const auto &act : actual){
            if(predicted.count(act.first) > 0){
                sq_err += std::pow(act.second - predicted[act.first], 2);
                ++n;
            }
        }
        if(n > 0)
            return std::sqrt(sq_err / n);
        else
            return -1;
    }
};


/******************
 *
 * RANKING METRICS
 *
 * ****************/

template<std::size_t N = 10, int RelTh = 4, typename Key = id_type>
struct Precision{
    static double eval(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        std::size_t rel_count{0};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        auto length = std::distance(ranking.cbegin(), it_end);
        for(auto it = ranking.cbegin(); it != it_end; ++it){
            if(relevance.count(*it) > 0 &&
                    relevance[*it] >= RelTh)
                ++rel_count;
        }
        return (double) rel_count / length;
    }
    static double eval_fast(const std::vector<Key> &ranking, __attribute__((unused))  const std::vector<Key> &best_ranking, hash_map_t<Key, double> &relevance){
        std::size_t rel_count{0};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        auto length = std::distance(ranking.cbegin(), it_end);
        for(auto it = ranking.cbegin(); it != it_end; ++it){
            if(relevance[*it] >= RelTh)
                ++rel_count;
        }
        return (double) rel_count / length;
    }
};

template<std::size_t N = 10, int RelTh = 4, typename Key = id_type>
struct AveragePrecision{
    static double eval(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        double p_at_k{.0};
        std::size_t num_relevant{0};
        for(const auto &entry : relevance)
            if(entry.second >= RelTh)
                ++num_relevant;
        if(num_relevant > 0){
            std::size_t rel_count{0}, rank{1};
            auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
            for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
                if(relevance.count(*it) > 0 &&
                        relevance[*it] >= RelTh)
                    p_at_k += (double) ++rel_count / rank;
            }
            return p_at_k / num_relevant;
        } else {
            return 0;
        }
    }
    static double eval_fast(const std::vector<Key> &ranking, __attribute__((unused))  const std::vector<Key> &best_ranking, hash_map_t<Key, double> &relevance){
        double p_at_k{.0};
        std::size_t num_relevant{0};
        for(const auto &entry : relevance)
            if(entry.second >= RelTh)
                ++num_relevant;
        if(num_relevant > 0){
            std::size_t rel_count{0}, rank{1};
            auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
            for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
                if(relevance[*it] >= RelTh)
                    p_at_k += (double) ++rel_count / rank;
            }
            return p_at_k / num_relevant;
        } else {
            return 0;
        }
    }
};

template<std::size_t N = 10, typename Key = id_type>
struct NDCG{
    static double eval(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        // best achievable ranking
        std::vector<Key> best_ranking(extract_keys(relevance));
        std::sort(best_ranking.begin(), best_ranking.end(),
                  [&](const Key &lhs, const Key &rhs){
            return relevance[lhs] > relevance[rhs];
        });
        return DCG(ranking, relevance) / DCG(best_ranking, relevance);
    }
    static double eval_fast(const std::vector<Key> &ranking, const std::vector<Key> &best_ranking, hash_map_t<Key, double> &relevance){
        return DCG_fast(ranking, relevance) / DCG_fast(best_ranking, relevance);
    }
private:
    static double DCG(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        double dcg{.0};
        std::size_t rank{1};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            if(relevance.count(*it) > 0)
                dcg += (std::pow(2, relevance[*it])-1) / std::log2(rank+1);
        }
        return dcg;
    }
    static double DCG_fast(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        double dcg{.0};
        std::size_t rank{1};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            dcg += (std::pow(2, relevance[*it])-1) / std::log2(rank+1);
        }
        return dcg;
    }
};

template<std::size_t N = 10, unsigned HL = 5, typename Key = id_type>
struct HLU{
    static double eval(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        // best achievable ranking
        std::vector<Key> best_ranking(extract_keys(relevance));
        std::sort(best_ranking.begin(), best_ranking.end(),
                  [&](const Key &lhs, const Key &rhs){
            return relevance[lhs] > relevance[rhs];
        });
        return _HLU(ranking, relevance) / _HLU(best_ranking, relevance);
    }
    static double eval_fast(const std::vector<Key> &ranking, const std::vector<Key> &best_ranking, hash_map_t<Key, double> &relevance){
        return _HLU(ranking, relevance) / _HLU(best_ranking, relevance);
    }

private:
    static double _HLU(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        double hlu{.0};
        std::size_t rank{1};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            if(relevance.count(*it) > 0)
                hlu += relevance[*it] / std::pow(2, (rank-1) / (HL-1));
        }
        return hlu;
    }
    static double _HLU_fast(const std::vector<Key> &ranking, hash_map_t<Key, double> &relevance){
        double hlu{.0};
        std::size_t rank{1};
        auto it_end = ranking.size() > N ? ranking.cbegin() + N : ranking.cend();
        for(auto it = ranking.cbegin(); it != it_end; ++it, ++rank){
            hlu += relevance[*it] / std::pow(2, (rank-1) / (HL-1));
        }
        return hlu;
    }
};


#endif // METRICS_HPP
