#ifndef RATING_HPP
#define RATING_HPP
#include <boost/fusion/adapted.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/spirit/include/qi.hpp>
#include <map>
#include <sstream>

namespace qi = boost::spirit::qi;

struct Rating{
    std::size_t _user_id;
    std::size_t _item_id;
    double _value;

    Rating(const std::size_t user_id, const std::size_t item_id, double value):
        _user_id{user_id}, _item_id{item_id}, _value{value}{}
    Rating() : _user_id{0}, _item_id{0}, _value{.0}{}
    Rating(const std::string &rating_str){
        std::istringstream iss(rating_str);
        iss >> _user_id;
        iss >> _item_id;
        iss >> _value;
    }
    friend std::ostream &operator <<(std::ostream &os, const Rating &t){
        os << "(" << t._user_id << ", " << t._item_id << "," << t._value << ")";
        return os;
    }

    static std::vector<Rating> read_from(const std::string &training_filename, const std::size_t sz_hint = 10000){
        boost::iostreams::mapped_file mmap(training_filename, boost::iostreams::mapped_file::readonly);
        auto f = mmap.const_data();
        auto l = f + mmap.size();

        std::vector<Rating> ratings;
        ratings.reserve(sz_hint);
        if(!qi::phrase_parse(f,l,(qi::ulong_long > qi::ulong_long > qi::double_) % qi::eol, qi::blank, ratings))
            throw std::runtime_error ("Unable to parse the training file.");
        return ratings;
    }

};
BOOST_FUSION_ADAPT_STRUCT(Rating, (std::size_t, _user_id)(std::size_t, _item_id)(double, _value))


#endif // RATING_HPP
