#ifndef AUX_HPP
#define AUX_HPP
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include <vector>

constexpr bool almost_eq(double lhs, double rhs, double eps = 1e-12) {
    return std::abs(lhs - rhs) < eps;
}

template<typename K, typename V>
std::vector<K> extract_keys(const std::map<K,V> &map){
    std::vector<K> keys;
    keys.reserve(map.size());
    std::for_each(map.cbegin(), map.cend(),[&](const std::pair<K,V> &entry){
        keys.push_back(entry.first);
    });
    return keys;
}

template<typename K, typename V>
std::vector<K> extract_keys(const std::unordered_map<K,V> &map){
    std::vector<K> keys;
    keys.reserve(map.size());
    std::for_each(map.cbegin(), map.cend(),[&](const std::pair<K,V> &entry){
        keys.push_back(entry.first);
    });
    return keys;
}

template<typename It>
bool is_ordered(It begin, It end){
    if(std::distance(begin, end) == 0) return true;
    for(auto it = begin; it != end-1; ++it)
        if(!(*it < *(it+1) || *it == *(it+1))) return false;
    return true;
}

template<typename It>
std::ostream& print_range(std::ostream &os, It begin, It end, const std::string delim = ", "){
    for(auto it = begin; it != end-1; ++it)
        os << *it << delim;
    os << *(end-1);
    return os;
}

#endif // AUX_HPP
