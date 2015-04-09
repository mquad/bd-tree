#include "metrics.hpp"
#include "rank_tree.hpp"
#include "stopwatch.hpp"
#include "d_tree_eval.hpp"
constexpr unsigned N = 50;
using PrecIndex = RankIndex<std::size_t, Precision<N>>;
using APIndex = RankIndex<std::size_t, AveragePrecision<N>>;
using NDCGIndex = RankIndex<std::size_t, NDCG<N>>;
using HLUIndex = RankIndex<std::size_t, HLU<N>>;

int main(int argc, char **argv)
{
    if(argc < 15){
        std::cout << "Usage: ./bdtree_rank <metric> <training-file> <validation-file> <query-file> <test-file> <lambda> <h-smooth> <max-depth> <min-ratings> <top-pop> <threads> <randomize> <rand-coeff> <size-hint> <outfile>" << std::endl;
        return 1;
    }
    std::string metric(argv[1]);
    std::string training_file(argv[2]);
    std::string validation_file(argv[3]);
    std::string query_file(argv[4]);
    std::string test_file(argv[5]);
    double lambda = std::strtod(argv[6], nullptr);
    double h_smoothing = std::strtod(argv[7], nullptr);
    unsigned max_depth = std::strtoul(argv[8], nullptr, 10);
    std::size_t min_ratings = std::strtoull(argv[9], nullptr, 10);
    std::size_t top_pop = std::strtoull(argv[10], nullptr, 10);
    unsigned num_threads = std::strtoul(argv[11], nullptr, 10);
    bool randomize = std::strtol(argv[12], nullptr, 10);
    double rand_coeff = std::strtod(argv[13], nullptr);
    std::size_t sz_hint = std::strtoull(argv[14], nullptr, 10);
    std::string outfile(argv[15]);

    std::ofstream ofs(outfile);

    stopwatch sw;
    sw.reset();
    sw.start();
    // build the decision tree
    std::unique_ptr<DTree<ABDNode>> bdtree;
    std::cout << "Ranking Metric: ";
    if(metric == "prec"){
        std::cout << "Precision@" << N << std::endl;
        bdtree = std::unique_ptr<DTree<ABDNode>>(new RankTree<PrecIndex>{lambda, h_smoothing, max_depth, min_ratings, top_pop, num_threads, randomize, rand_coeff});
    }else if(metric == "ap"){
        std::cout << "AP@" << N << std::endl;
        bdtree = std::unique_ptr<DTree<ABDNode>>(new RankTree<APIndex>{lambda, h_smoothing, max_depth, min_ratings, top_pop, num_threads, randomize, rand_coeff});
    }else if(metric == "ndcg"){
        std::cout << "NDCG@" << N << std::endl;
        bdtree = std::unique_ptr<DTree<ABDNode>>(new RankTree<NDCGIndex>{lambda, h_smoothing, max_depth, min_ratings, top_pop, num_threads, randomize, rand_coeff});
    }else if(metric == "hlu"){
        std::cout << "HLU@" << N << std::endl;
        bdtree = std::unique_ptr<DTree<ABDNode>>(new RankTree<HLUIndex>{lambda, h_smoothing, max_depth, min_ratings, top_pop, num_threads, randomize, rand_coeff});
    }else{
        std::cerr << "Unknown metric. Valid values are: prec, ap, ndcg, hlu." << std::endl;
    }
    bdtree->init(Rating::read_from(training_file, sz_hint));
    auto init_t = sw.elapsed_ms();
    std::cout << "Tree initialized in " << init_t / 1000.0 << " s." << std::endl ;
    bdtree->build();
    std::cout << "Tree built in " << (sw.elapsed_ms() - init_t) / 1000.0 << " s." << std::endl ;

    user_profiles_t query_profiles, test_profiles;
    build_profiles(query_file, query_profiles);
    build_profiles(test_file, test_profiles);
    // evaluate tree quality
    auto rmse = evaluate_error<decltype(*bdtree), RMSE<>>(*bdtree, query_profiles, test_profiles);
    ofs << "RMSE=["; print_range(ofs, rmse.cbegin(), rmse.cend()) << "]" << std::endl;
    auto p = evaluate_ranking<decltype(*bdtree), Precision<N>>(*bdtree, query_profiles, test_profiles);
    ofs << "Precision@" << N << "=["; print_range(ofs, p.cbegin(), p.cend()) << "]" << std::endl;
    auto map = evaluate_ranking<decltype(*bdtree), AveragePrecision<N>>(*bdtree, query_profiles, test_profiles);
    ofs << "MAP@" << N << "=["; print_range(ofs, map.cbegin(), map.cend()) << "]" << std::endl;
    auto ndcg = evaluate_ranking<decltype(*bdtree), NDCG<N>>(*bdtree, query_profiles, test_profiles);
    ofs << "NDCG@" << N << "=["; print_range(ofs, ndcg.cbegin(), ndcg.cend()) << "]" << std::endl;
    auto hlu = evaluate_ranking<decltype(*bdtree), HLU<N,5>>(*bdtree, query_profiles, test_profiles);
    ofs << "HLU@" << N << "=["; print_range(ofs, hlu.cbegin(), hlu.cend()) << "]" << std::endl;
    std::cout << "Process completed in " << sw.elapsed_ms() / 1000.0  << " s." << std::endl;
    return 0;
}
