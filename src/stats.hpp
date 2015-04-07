#ifndef STATS_HPP
#define STATS_HPP
#include <iostream>
#include <map>
#include "score.hpp"

template<typename S>
struct Stats{
    using score_t = S;
    // the minimal statistics is made of the number of ratings per item
    int _n;
    Stats(int n) : _n{n}{}
    Stats() : _n{0}{}

    virtual void update(const S &score) = 0;
    virtual double quality() const = 0;
};

struct ABDStats : public Stats<ScoreUnbiased>{
    using Stats<ScoreUnbiased>::_n;
    double _sum, _sum_unbiased;
    double _sum2, _sum2_unbiased;

    ABDStats(double sum, double sum_unbiased, double sum2, double sum2_unbiased, int n) :
        Stats<ScoreUnbiased>(n),
        _sum{sum}, _sum_unbiased{sum_unbiased}, _sum2{sum2}, _sum2_unbiased{sum2_unbiased}{}
    ABDStats() : ABDStats(.0, .0, .0, .0, 0){}

    void update(const ScoreUnbiased &score){
        _sum += score._rating;
        _sum_unbiased += score._rating_unbiased;
        _sum2 += score._rating * score._rating;
        _sum2_unbiased += score._rating_unbiased * score._rating_unbiased;
        ++_n;
    }

    double quality() const override{
        if(_n <= 0) throw std::runtime_error("n <= 0");
        return -(_sum2_unbiased - (_sum_unbiased * _sum_unbiased) / _n);
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
        q += entry.second.quality();
    return q;
}
#endif // STATS_HPP
