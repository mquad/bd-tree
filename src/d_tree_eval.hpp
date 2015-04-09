#ifndef EVALUATION_HPP
#define EVALUATION_HPP

#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include "d_tree.hpp"
#include "metrics.hpp"

using user_profiles_t = std::map<unsigned long, profile_t>;

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

template<typename T, typename Metric>
std::vector<double> evaluate_error(const T &dtree,
                                     const user_profiles_t &answers,
                                     const user_profiles_t &test){
    std::vector<double> metric_avg(dtree.depth_max(), .0);
    std::vector<int> counts(dtree.depth_max(), 0);
    for(const auto &ans : answers){
        if(test.count(ans.first) > 0){
            unsigned level{0u};
            auto node = dtree.root();
            while(node != nullptr){
                profile_t predicted;
                for(const auto &t : test.at(ans.first)){
                    try{
                        predicted.emplace(t.first, dtree.predict(node, t.first));
                    }catch(std::out_of_range &){}
                }
                metric_avg[level] += Metric::eval(predicted, test.at(ans.first));
                ++counts[level];
                node = dtree.traverse(node, ans.second);
                ++level;
            }
        }
    }
    std::transform(metric_avg.begin(), metric_avg.end(),
                   counts.begin(), metric_avg.begin(),
                   [](const double &sum, const int &count){return sum / count;});
    return metric_avg;
}

template<typename T, typename Metric, int RelTh = 4>
std::vector<double> evaluate_ranking(const T &dtree,
                                     const user_profiles_t &answers,
                                     const user_profiles_t &test){
    std::vector<double> metric_avg(dtree.depth_max(), .0);
    std::vector<int> counts(dtree.depth_max(), 0);
    for(const auto &ans : answers){
        if(test.count(ans.first) > 0){
            std::unordered_set<unsigned long> relevant_test;
            for(const auto &entry : test.at(ans.first))
                if(entry.second >= RelTh)   relevant_test.insert(entry.first);
            unsigned level{0u};
            auto node = dtree.root();
            while(node != nullptr){
                metric_avg[level] += Metric::eval(dtree.ranking(node), relevant_test);
                ++counts[level];
                node = dtree.traverse(node, ans.second);
                ++level;
            }
        }
    }
    std::transform(metric_avg.begin(), metric_avg.end(),
                   counts.begin(), metric_avg.begin(),
                   [](const double &sum, const int &count){return sum / count;});
    return metric_avg;
}

#endif // EVALUATION_HPP
