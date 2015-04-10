#ifndef STATS_HPP
#define STATS_HPP
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include "score.hpp"

struct ABDStats{
    using score_t = ScoreUnbiased;

    double _sum, _sum_unbiased;
    double _sum2, _sum2_unbiased;
    int _n;

    ABDStats(double sum, double sum_unbiased, double sum2, double sum2_unbiased, int n) :
        _sum{sum}, _sum_unbiased{sum_unbiased}, _sum2{sum2}, _sum2_unbiased{sum2_unbiased}, _n{n}{}
    ABDStats() : ABDStats(.0, .0, .0, .0, 0){}

    void update(const ScoreUnbiased &score){
        _sum += score._rating;
        _sum_unbiased += score._rating_unbiased;
        _sum2 += score._rating * score._rating;
        _sum2_unbiased += score._rating_unbiased * score._rating_unbiased;
        ++_n;
    }

    double squared_error() const{
        if(_n <= 0) throw std::runtime_error("n <= 0");
        return _sum2_unbiased - (_sum_unbiased * _sum_unbiased) / _n;
    }

    double pred() const{
        if(_n <= 0) throw std::runtime_error("n <= 0");
        return _sum / _n;
    }

    double pred(const double parent_pred, const double h_smooth) const{
        return (_sum + h_smooth * parent_pred) / (_n + h_smooth);
    }

    double score() const{
        if(_n <= 0) throw std::runtime_error("n <= 0");
        return _sum_unbiased / _n;
    }

    double score(const double parent_score, const double h_smooth) const{
        return (_sum_unbiased + h_smooth * parent_score) / (_n + h_smooth);
    }

    ABDStats& operator-=(const ABDStats &rhs){
        this->_sum -= rhs._sum;
        this->_sum_unbiased -= rhs._sum_unbiased;
        this->_sum2 -= rhs._sum2;
        this->_sum2_unbiased -= rhs._sum2_unbiased;
        this->_n -= rhs._n;
        return *this;
    }

    friend std::ostream& operator<< (std::ostream &os, const ABDStats &stats){
        os << "sum: " << stats._sum
              << "\tsum(unbiased): " << stats._sum_unbiased
              << "\tsum2: " << stats._sum2
              << "\tsum2(unbiased): " << stats._sum2_unbiased
              << "\tn: " << stats._n;
        return os;
    }
};

template<typename K, typename S>
using StatMap = std::map<K, S>;

template<typename K, typename S>
double compute_quality(const StatMap<K, S> &map){
    double q{.0};
    for(const auto entry : map)
        q -= entry.second.squared_error();
    return q;
}


template<typename K, typename S>
std::vector<K> build_ranking(const StatMap<K, S> &stats){
    std::vector<std::pair<K, double>> items_by_score;
    for(const auto &entry : stats)
        items_by_score.emplace_back(entry.first, entry.second.score());
    std::sort(items_by_score.begin(),
              items_by_score.end(),
              [](const std::pair<K, double> &lhs, const std::pair<K, double> &rhs){
        return lhs.second > rhs.second;
    });
    std::vector<K> ranking;
    ranking.reserve(items_by_score.size());
    for(const auto &item : items_by_score)
        ranking.emplace_back(item.first);
    return ranking;
}


template<typename K, typename S>
std::vector<K> build_ranking(const StatMap<K, S> &stats, const std::map<K, double> &parent_scores, const double h_smooth){
    // sort items by smoothed score
    std::vector<std::pair<K, double>> items_by_score;
    items_by_score.reserve(parent_scores.size());
    for(const auto &p_pred : parent_scores){
        if(stats.count(p_pred.first) > 0){
            items_by_score.emplace_back(p_pred.first, stats.at(p_pred.first).score(p_pred.second, h_smooth));
        }else{
            items_by_score.emplace_back(p_pred.first, p_pred.second);
        }
    }
    std::sort(items_by_score.begin(),
              items_by_score.end(),
              [](const std::pair<K, double> &lhs, const std::pair<K, double> &rhs){
        return lhs.second > rhs.second;
    });
    std::vector<K> ranking;
    ranking.reserve(items_by_score.size());
    for(const auto &item : items_by_score)
        ranking.emplace_back(item.first);
    return ranking;
}

#endif // STATS_HPP
