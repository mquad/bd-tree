#include "metrics.hpp"
#include "rank_tree.hpp"
#include "stopwatch.hpp"
#include "d_tree_eval.hpp"

constexpr unsigned N = 10;
using PrecIndex = RankIndex<id_type, Precision<N>>;
using APIndex = RankIndex<id_type, AveragePrecision<N>>;
using NDCGIndex = RankIndex<id_type, NDCG<N>>;
using HLUIndex = RankIndex<id_type, HLU<N>>;


void print_usage_build(){
    std::cout << "BUILD ONLY (no prediction / evaluation):" << std::endl
              << "Usage: ./bdtree_rank build <metric> <training-file> <lambda> <h-smooth> <max-depth> <min-ratings> <top-pop> <threads> <randomize> <rand-coeff>" << std::endl;
}

void print_usage_eval(){
    std::cout << "PREDICTION / EVALUATION" << std::endl
              << "Usage: ./bdtree_rank eval <metric> <training-file> <answer-file> <evaluation-file> <lambda> <h-smooth> <max-depth> <min-ratings> <top-pop> <threads> <randomize> <rand-coeff> <outfile>" << std::endl;
}

int main(int argc, char **argv)
{
    if(argc < 2 || std::string(argv[1]) == "help"){
        print_usage_build(); print_usage_eval();
        return 1;
    }
    std::string mode(argv[1]);
    if(mode == "build"){
        if(argc < 12){
            print_usage_build();
            return 1;
        }
        std::string metric(argv[2]);
        std::string training_file(argv[3]);
        double lambda = std::strtod(argv[4], nullptr);
        double h_smoothing = std::strtod(argv[5], nullptr);
        unsigned max_depth = std::strtoul(argv[6], nullptr, 10);
        std::size_t min_ratings = std::strtoull(argv[7], nullptr, 10);
        std::size_t top_pop = std::strtoull(argv[8], nullptr, 10);
        unsigned num_threads = std::strtoul(argv[9], nullptr, 10);
        bool randomize = std::strtol(argv[10], nullptr, 10);
        double rand_coeff = std::strtod(argv[11], nullptr);

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

        bdtree->init(Rating::read_from(training_file));
        auto init_t = sw.elapsed_ms();
        std::cout << "Tree initialized in " << init_t / 1000.0 << " s." << std::endl ;
        bdtree->build();
        std::cout << "Tree built in " << (sw.elapsed_ms() - init_t) / 1000.0 << " s." << std::endl ;
        bdtree->release_temp();
        std::cout << "Temporaries released." << (sw.elapsed_ms() - init_t) / 1000.0 << " s." << std::endl ;

        return 0;
    }else if (mode == "eval"){
        if(argc < 15){
            print_usage_eval();
            return 1;
        }
        std::string metric(argv[2]);
        std::string training_file(argv[3]);
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
        std::string outfile(argv[14]);

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

        bdtree->init(Rating::read_from(training_file));
        auto init_t = sw.elapsed_ms();
        std::cout << "Tree initialized in " << init_t / 1000.0 << " s." << std::endl ;
        bdtree->build();
        std::cout << "Tree built in " << (sw.elapsed_ms() - init_t) / 1000.0 << " s." << std::endl ;

        user_profiles_t query_profiles, test_profiles;
        build_profiles(query_file, query_profiles);
        build_profiles(test_file, test_profiles);
        // evaluate tree quality
        std::cout << "EVALUATION:" << std::endl
                  << "Num. test users: " << test_profiles.size() << std::endl;
        auto added = added_ratings<decltype(*bdtree)>(*bdtree, query_profiles, test_profiles);
        ofs << "ADDED=["; print_range(ofs, added.cbegin(), added.cend()) << "]" << std::endl;
        auto rmse = evaluate_rmse<decltype(*bdtree)>(*bdtree, query_profiles, test_profiles);
        ofs << "RMSE=["; print_range(ofs, rmse.cbegin(), rmse.cend()) << "]" << std::endl;
        auto p = evaluate_ranking<decltype(*bdtree), Precision<N>>(*bdtree, query_profiles, test_profiles);
        ofs << "Precision_at_" << N << "=["; print_range(ofs, p.cbegin(), p.cend()) << "]" << std::endl;
        auto map = evaluate_ranking<decltype(*bdtree), AveragePrecision<N>>(*bdtree, query_profiles, test_profiles);
        ofs << "MAP_at_" << N << "=["; print_range(ofs, map.cbegin(), map.cend()) << "]" << std::endl;
        auto ndcg = evaluate_ranking<decltype(*bdtree), NDCG<N>>(*bdtree, query_profiles, test_profiles);
        ofs << "NDCG_at_" << N << "=["; print_range(ofs, ndcg.cbegin(), ndcg.cend()) << "]" << std::endl;
        auto hlu = evaluate_ranking<decltype(*bdtree), HLU<N,5>>(*bdtree, query_profiles, test_profiles);
        ofs << "HLU_at_" << N << "=["; print_range(ofs, hlu.cbegin(), hlu.cend()) << "]" << std::endl;
        std::cout << "Process completed in " << sw.elapsed_ms() / 1000.0  << " s." << std::endl;
        return 0;
    }else{
        std::cerr << "Unsupported mode: " << mode << std::endl;
        return 1;
    }
}
