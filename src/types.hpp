#ifndef RATING_HPP
#define RATING_HPP

#include <sstream>
#include <boost/fusion/adapted.hpp>

struct rating_t{
    std::size_t _user_id;
    std::size_t _item_id;
    double _value;
public:
    rating_t(const std::size_t user_id, const std::size_t item_id, double value):
        _user_id{user_id}, _item_id{item_id}, _value{value}{}
    rating_t() : _user_id{0}, _item_id{0}, _value{.0}{}
    rating_t(const std::string &rating_str){
        std::istringstream iss(rating_str);
        iss >> _user_id;
        iss >> _item_id;
        iss >> _value;
    }
    friend std::ostream &operator <<(std::ostream &os, const rating_t &t){
        os << "(" << t._user_id << ", " << t._item_id << "," << t._value << ")";
        return os;
    }

};
BOOST_FUSION_ADAPT_STRUCT(rating_t, (std::size_t, _user_id)(std::size_t, _item_id)(double, _value))

using profile_t = std::unordered_map<std::size_t, double>;


#endif // RATING_HPP
