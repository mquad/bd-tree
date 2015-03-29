#include <iostream>
#include <fstream>
#include "bd_tree.hpp"
#include "evaluation.hpp"

std::vector<rating_t> import(const std::string &filename){
    std::ifstream ifs(filename);
    std::string line;
    std::vector<rating_t> ratings;
    while(std::getline(ifs, line))
        ratings.push_back(rating_t(line));
    return ratings;
}


int main(int argc, char **argv)
{
    if(argc < 9){
        std::cout << "Usage: ./bd-tree <training-file> <validation-file> <query-file> <test-file> <lambda> <h-smooth> <max-depth> <min-ratings>" << std::endl;
        return 1;
    }
    std::string training_file(argv[1]);
    std::string validation_file(argv[2]);
    std::string query_file(argv[3]);
    std::string test_file(argv[4]);
    double lambda = std::strtod(argv[5], nullptr);
    double h_smoothing = std::strtod(argv[6], nullptr);
    unsigned max_depth = std::strtoul(argv[7], nullptr, 10);
    std::size_t min_ratings = std::strtoull(argv[8], nullptr, 10);

    // build the decision tree
    BDTree bdtree{true, lambda, h_smoothing};
    bdtree.init(import(training_file));
    bdtree.build(max_depth, min_ratings);

    user_profiles_t query_profiles, test_profiles;
    double query_mu = build_profiles(query_file, query_profiles);
    build_profiles(test_file, test_profiles);
    auto query_bu = compute_bu(query_profiles, query_mu, lambda);
    // evaluate tree quality
    double rmse_val = evaluate(bdtree, query_profiles, test_profiles, query_bu, rmse);
    std::cout << "RMSE: " << rmse_val << std::endl;
    return 0;
}

