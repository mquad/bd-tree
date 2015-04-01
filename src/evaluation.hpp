#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "bd_tree.hpp"

using user_profiles_t = std::unordered_map<std::size_t, profile_t>;

double build_profiles(const std::string &filename, user_profiles_t &profiles){
    std::ifstream ifs(filename);
    std::string line;
    double sum{};
    std::size_t n{};
    while(std::getline(ifs, line)){
        std::istringstream iss(line);
        std::size_t user_id, item_id;
        double rating;
        iss >> user_id;
        iss >> item_id;
        iss >> rating;
        profiles[user_id][item_id] = rating;
        sum += rating;
        ++n;
    }
    return sum / n;
}

double rmse(const BDTree &bdtree, const profile_t &answers, const profile_t &test){
    double mse{};
    std::size_t n{0};
    const auto leaf = bdtree.traverse(answers);
    for(const auto &ans : test){
        try{
            double pred_r = bdtree.predict(leaf, ans.first);
            double actual_r = ans.second;
            mse += std::pow(pred_r - actual_r, 2);
        }catch(std::out_of_range &){
            std::cout << "unable to predict for item " << ans.first << std::endl;
        }
    }
    return std::sqrt(mse / n);
}

double evaluate(const BDTree &bdtree,
                const user_profiles_t &query,
                const user_profiles_t &test,
                double (*metric)(const BDTree&, const profile_t&, const profile_t&)){
    double metric_sum{};
    std::size_t n{query.size()};
    for(const auto &ans : query){
        if(test.count(ans.first) > 0)
            metric_sum += metric(bdtree, ans.second, test.at(ans.first));
    }
    return metric_sum / n;

}


#endif // EVALUATION_HPP
