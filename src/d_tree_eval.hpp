#ifndef EVALUATION_HPP
#define EVALUATION_HPP
#include <fstream>
#include <sstream>
#include <unordered_set>
#include "aux.hpp"
#include "d_tree.hpp"
#include "metrics.hpp"

using user_profiles_t = std::map<id_type, profile_t>;

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
        if(profiles.count(user_id) == 0)
            profiles[user_id].set_empty_key(-1);
        profiles[user_id][item_id] = rating;
        sum += rating;
        ++n;
    }
    return sum / n;
}


template<typename T>
std::vector<double> added_ratings(const T &dtree,
                                   user_profiles_t &answers,
                                   user_profiles_t &test){
    std::vector<double> metric_avg(dtree.depth_max(), .0);
    std::vector<int> counts(dtree.depth_max(), 0);
    for(const auto &ans : answers){
        if(test.count(ans.first) > 0){
            unsigned level{0u};
            auto node = dtree.root();
            while(!node->is_leaf()){
                int added = dtree.traverse(node, ans.second);
                metric_avg[level] += static_cast<double>(added);
                ++counts[level];
                ++level;
            }
        }
    }
    std::transform(metric_avg.begin(), metric_avg.end(),
                   counts.begin(), metric_avg.begin(),
                   [](const double &sum, const int &count){return count > 0 ? sum / count : -1;});
    // remove trailing elements
    auto it = metric_avg.begin();
    while(it != metric_avg.end() && *it != -1) ++it;
    metric_avg.resize(std::distance(metric_avg.begin(), it));
    return metric_avg;
}

template<typename T, typename Metric>
std::vector<double> evaluate_error(const T &dtree,
                                     user_profiles_t &answers,
                                     user_profiles_t &test){
    std::vector<double> metric_avg(dtree.depth_max(), .0);
    std::vector<int> counts(dtree.depth_max(), 0);
    for(const auto &ans : answers){
        if(test.count(ans.first) > 0){
            auto test_ids = extract_keys(test.at(ans.first));
            unsigned level{0u};
            auto node = dtree.root();
            while(node != nullptr){
                profile_t predicted = dtree.predict(node, test_ids);
                double metric = Metric::eval(predicted, test.at(ans.first));
                if(metric != -1){
                    metric_avg[level] += metric;
                    ++counts[level];
                }
                dtree.traverse(node, ans.second);
                ++level;
            }
        }
    }
    std::transform(metric_avg.begin(), metric_avg.end(),
                   counts.begin(), metric_avg.begin(),
                   [](const double &sum, const int &count){return count > 0 ? sum / count : -1;});
    // remove trailing elements
    auto it = metric_avg.begin();
    while(it != metric_avg.end() && *it != -1) ++it;
    metric_avg.resize(std::distance(metric_avg.begin(), it));
    return metric_avg;
}

template<typename T, typename Metric, int RelTh = 4>
std::vector<double> evaluate_ranking(const T &dtree,
                                     user_profiles_t &answers,
                                     user_profiles_t &test){
    std::vector<double> metric_avg(dtree.depth_max(), .0);
    std::vector<int> counts(dtree.depth_max(), 0);
    for(const auto &ans : answers){
        if(test.count(ans.first) > 0){
            auto test_ids = extract_keys(test.at(ans.first));
            unsigned level{0u};
            auto node = dtree.root();
            while(node != nullptr){
                metric_avg[level] += Metric::eval(dtree.ranking(node, test_ids), test.at(ans.first));
                ++counts[level];
                dtree.traverse(node, ans.second);
                ++level;
            }
        }
    }
    std::transform(metric_avg.begin(), metric_avg.end(),
                   counts.begin(), metric_avg.begin(),
                   [](const double &sum, const int &count){return count > 0 ? sum / count : -1;});
    // remove trailing elements
    auto it = metric_avg.begin();
    while(it != metric_avg.end() && *it != -1) ++it;
    metric_avg.resize(std::distance(metric_avg.begin(), it));
    return metric_avg;
}

#endif // EVALUATION_HPP
