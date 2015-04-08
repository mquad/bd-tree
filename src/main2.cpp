#include "abd_tree.hpp"
#include "stopwatch.hpp"
#include "d_tree_eval.hpp"

int main(int argc, char **argv)
{
    if(argc < 14){
        std::cout << "Usage: ./bd-tree2 <training-file> <validation-file> <query-file> <test-file> <lambda> <h-smooth> <max-depth> <min-ratings> <top-pop> <threads> <randomize> <rand-coeff> <size-hint>" << std::endl;
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
    std::size_t top_pop = std::strtoull(argv[9], nullptr, 10);
    unsigned num_threads = std::strtoul(argv[10], nullptr, 10);
    bool randomize = std::strtol(argv[11], nullptr, 10);
    double rand_coeff = std::strtod(argv[12], nullptr);
    std::size_t sz_hint = std::strtoull(argv[13], nullptr, 10);



    stopwatch sw;
    sw.reset();
    sw.start();

    // build the decision tree
    ABDTree<ABDNode> bdtree{lambda, h_smoothing, max_depth, min_ratings, top_pop, num_threads, randomize, rand_coeff};
    bdtree.init(Rating::read_from(training_file, sz_hint));
    auto init_t = sw.elapsed_ms();
    std::cout << "Tree initialized in " << init_t / 1000.0 << " s." << std::endl ;
    bdtree.build();
    std::cout << "Tree built in " << (sw.elapsed_ms() - init_t) / 1000.0 << " s." << std::endl ;

    user_profiles_t query_profiles, test_profiles;
    build_profiles(query_file, query_profiles);
    build_profiles(test_file, test_profiles);
    // evaluate tree quality
    double rmse_val = evaluate<decltype(bdtree)>(bdtree, query_profiles, test_profiles, rmse);
    std::cout << "RMSE: " << rmse_val << std::endl;

    double map = evaluate_ranking<decltype(bdtree), AveragePrecision<50>>(bdtree, query_profiles, test_profiles);
    std::cout << "MAP: " << map << std::endl;
    double ndcg = evaluate_ranking<decltype(bdtree), NDCG<50>>(bdtree, query_profiles, test_profiles);
    std::cout << "NDCG: " << ndcg << std::endl;
    std::cout << "Process completed in " << sw.elapsed_ms() / 1000.0  << " s." << std::endl;
    return 0;
}
