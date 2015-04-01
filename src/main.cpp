#include <iostream>
#include <fstream>
#include "bd_tree.hpp"
#include "bd_tree_eval.hpp"
#include "stopwatch.hpp"

#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

namespace qi = boost::spirit::qi;


std::vector<rating_t> import(const std::string &filename){
    std::ifstream ifs(filename);
    std::string line;
    std::vector<rating_t> ratings;
    while(std::getline(ifs, line))
        ratings.push_back(rating_t(line));
    return ratings;
}

std::vector<rating_t> import2(const std::string &filename){
    boost::iostreams::mapped_file mmap(filename, boost::iostreams::mapped_file::readonly);
    auto f = mmap.const_data();
    auto l = f + mmap.size();

    std::vector<rating_t> ratings;
    ratings.reserve(10000000);
    bool ok = qi::phrase_parse(f,l,(qi::ulong_long > qi::ulong_long > qi::double_) % qi::eol, qi::blank, ratings);
    if (ok)
        std::cout << "parse success\n";
    else
        std::cerr << "parse failed: '" << std::string(f,l) << "'\n";
    return ratings;
}


int main(int argc, char **argv)
{
    if(argc < 10){
        std::cout << "Usage: ./bd-tree <training-file> <validation-file> <query-file> <test-file> <lambda> <h-smooth> <max-depth> <min-ratings> <threads>" << std::endl;
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
    unsigned num_threads = std::strtoul(argv[9], nullptr, 10);


    stopwatch sw;
    sw.reset();
    sw.start();
    // build the decision tree
    BDTree bdtree{lambda, h_smoothing, num_threads};
    bdtree.init(import2(training_file));
    auto init_t = sw.elapsed_ms();
    std::cout << "Tree initialized in " << init_t / 1000.0 << " s." << std::endl ;
//    bdtree.build(max_depth, min_ratings);
//    std::cout << "Tree built in " << (sw.elapsed_ms() - init_t) / 1000.0 << " s." << std::endl ;

//    user_profiles_t query_profiles, test_profiles;
//    build_profiles(query_file, query_profiles);
//    build_profiles(test_file, test_profiles);
//    // evaluate tree quality
//    double rmse_val = evaluate(bdtree, query_profiles, test_profiles, rmse);
//    std::cout << "RMSE: " << rmse_val << std::endl;
//    std::cout << "Process completed in " << sw.elapsed_ms() / 1000.0  << " s." << std::endl;
//    return 0;
}

