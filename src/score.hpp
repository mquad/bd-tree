#ifndef SCORE_HPP
#define SCORE_HPP
#include <iostream>
#include "aux.hpp"

struct ScoreUnbiased{
    std::size_t _id;
    double _rating;
    double _rating_unbiased;

    ScoreUnbiased(std::size_t id, double rating, double rating_unbiased) :
        _id{id}, _rating{rating}, _rating_unbiased{rating_unbiased}{}
    ScoreUnbiased() : ScoreUnbiased(0u, .0, .0){}

    friend bool operator ==(const ScoreUnbiased &lhs, const ScoreUnbiased &rhs){
        return (lhs._id == rhs._id) &&
                (almost_eq(lhs._rating, rhs._rating)) &&
                (almost_eq(lhs._rating_unbiased, rhs._rating_unbiased));
    }
    friend bool operator !=(const ScoreUnbiased &lhs, const ScoreUnbiased &rhs){
        return !(lhs == rhs);
    }
    friend std::ostream &operator <<(std::ostream &os, const ScoreUnbiased &s){
        os << "(" << s._id << ", "
           << s._rating << ", "
           << s._rating_unbiased << ")";
        return os;
    }
    friend bool operator< (const ScoreUnbiased &lhs, const ScoreUnbiased &rhs){
        return lhs._id < rhs._id ||
                (lhs._id == rhs._id && lhs._rating < rhs._rating);
    }

};


#endif // SCORE_HPP
