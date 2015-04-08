#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "d_tree.hpp"
#include "metrics.hpp"

using user_profiles_t = std::unordered_map<unsigned long, profile_t>;

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

template<typename T>
double rmse(const T &dtree, const profile_t &answers, const profile_t &test){
    double mse{0};
    std::size_t n{0};
    const auto leaf = dtree.traverse(answers);
    for(const auto &ans : test){
        try{
            double pred_r = dtree.predict(leaf, ans.first);
            double actual_r = ans.second;
            mse += std::pow(pred_r - actual_r, 2);
            ++n;
        }catch(std::out_of_range &){
        }
    }
    return std::sqrt(mse / n);
}

template<typename T>
double evaluate(const T &dtree,
                const user_profiles_t &query,
                const user_profiles_t &test,
                double (*metric)(const T&, const profile_t&, const profile_t&)){
    double metric_sum{.0};
    std::size_t n{0};
    for(const auto &ans : query){
        if(test.count(ans.first) > 0){
            metric_sum += metric(dtree, ans.second, test.at(ans.first));
            ++n;
        }
    }
    return metric_sum / n;
}

template<typename T, typename Metric, int RelTh = 4>
double evaluate_ranking(const T &dtree,
                        const user_profiles_t &query,
                        const user_profiles_t &test){
    double metric_sum{.0};
    int n{0};
    for(const auto &ans : query){
        if(test.count(ans.first) > 0){
            std::unordered_set<unsigned long> relevant_test;
            for(const auto &entry : test.at(ans.first))
                if(entry.second >= RelTh)   relevant_test.insert(entry.first);
            const auto leaf = dtree.traverse(ans.second);
            metric_sum += Metric::eval(dtree.ranking(leaf), relevant_test);
            ++n;
        }
    }
    return metric_sum / n;
}

#endif // EVALUATION_HPP
