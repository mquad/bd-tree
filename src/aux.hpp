#ifndef AUX_HPP
#define AUX_HPP
#include <cmath>
constexpr bool almost_eq(double lhs, double rhs, double eps = 1e-12) {
    return std::abs(lhs - rhs) < eps;
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
