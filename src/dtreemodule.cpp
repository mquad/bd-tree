#ifndef DTREEMODULE_CPP
#define DTREEMODULE_CPP

#include "abd_tree.hpp"
#include "rank_tree.hpp"
#include "metrics.hpp"
#include <boost/python.hpp>

namespace py = boost::python;

constexpr int N = 10;
using NDCGIndex = RankIndex<std::size_t, NDCG<N>>;

class ErrorTreePy : public ABDTree{
public:
    ErrorTreePy(const double bu_reg = 7,
                const double h_smooth = 100,
                const unsigned depth_max = 6,
                const std::size_t ratings_min = 200000,
                const std::size_t top_pop = 0,
                const unsigned num_threads = 1,
                const bool randomize = false,
                const double rand_coeff = 10) :
        ABDTree(bu_reg, h_smooth, depth_max, ratings_min, top_pop, num_threads, randomize, rand_coeff){}

    void init_py(const py::list &training){
        std::vector<Rating> training_data;
        long int size{py::len(training)};
        training_data.reserve(size);
        for(long int t{0u}; t < size; ++t){
            py::tuple rating = py::extract<py::tuple>(training[t]);
            training_data.push_back(Rating(
                                        py::extract<std::size_t>(rating[0]),
                                        py::extract<std::size_t>(rating[1]),
                                        py::extract<double>(rating[2])
                                        ));
        }
        this->init(training_data);
    }

    void build_py(const py::list &candidates){
        std::vector<std::size_t> candidates_vec;
        long int size{py::len(candidates)};
        candidates_vec.reserve(size);
        for(long int t{0}; t < size; ++t){
            candidates_vec.push_back(py::extract<std::size_t>(candidates[t]));
        }
        this->build(candidates_vec);
    }
};

template<typename R>
class RankTreePy : public RankTree<R>{
public:
    RankTreePy(const double bu_reg = 7,
               const double h_smooth = 100,
               const unsigned depth_max = 6,
               const std::size_t ratings_min = 200000,
               const std::size_t top_pop = 0,
               const unsigned num_threads = 1,
               const bool randomize = false,
               const double rand_coeff = 10) :
        RankTree<R>(bu_reg, h_smooth, depth_max, ratings_min, top_pop, num_threads, randomize, rand_coeff){}

    void init_py(const py::list &training){
        std::vector<Rating> training_data;
        training_data.reserve(py::len(training));
        for(int t{}; t < py::len(training); ++t){
            py::tuple rating = py::extract<py::tuple>(training[t]);
            training_data.push_back(Rating(
                                        py::extract<std::size_t>(rating[0]),
                                        py::extract<std::size_t>(rating[1]),
                                        py::extract<double>(rating[2])
                    ));
        }
        this->init(training_data);
    }

    void build_py(const py::list &candidates){
        std::vector<std::size_t> candidates_vec;
        long int size{py::len(candidates)};
        candidates_vec.reserve(size);
        for(long int t{0}; t < size; ++t){
            candidates_vec.push_back(py::extract<std::size_t>(candidates[t]));
        }
        this->build(candidates_vec);
    }

};

template<typename C, typename X1>
void expose_tree_methods(py::class_<C, X1> c) {
  c.def("init", &C::init_py)
          .def("build", &C::build_py);
}

template<typename T>
class TraverserPy{
    using node_t = typename T::node_cptr_t;
    using id_t = typename T::id_t;
    node_t _current_node;
public:
    TraverserPy(const T &tree) : _current_node{tree.root()} {}
    id_t current_query() const{
        return _current_node->_splitter_id;
    }
    bool at_leaf() const{
        return _current_node->is_leaf();
    }
    void traverse_loved(){
        if(_current_node->has_loved())
            _current_node = _current_node->traverse_loved();
    }
    void traverse_hated(){
        if(_current_node->has_hated())
            _current_node = _current_node->traverse_hated();
    }
    void traverse_unknown(){
        if(_current_node->has_unknown())
            _current_node = _current_node->traverse_unknown();
    }
};

template<typename T>
TraverserPy<T>* makeTraverser(const T &tree){
    return new TraverserPy<T>(tree);
}

template<typename T>
void expose_Traverser(const std::string &classname){
    using C = TraverserPy<T>;
    py::class_<C> c(classname.c_str(), py::no_init);
    c.def("__init__", py::make_constructor(&makeTraverser<T>))
            .def("current_query", &C::current_query)
            .def("at_leaf", &C::at_leaf)
            .def("traverse_loved", &C::traverse_loved)
            .def("traverse_hated", &C::traverse_hated)
            .def("traverse_unknown", &C::traverse_unknown);
}

BOOST_PYTHON_MODULE(dtreelib)
{
  expose_tree_methods(py::class_<ErrorTreePy, boost::noncopyable>("ErrorTreePy", py::init<double, double, unsigned, std::size_t, std::size_t, unsigned, bool, double>()));
  expose_tree_methods(py::class_<RankTreePy<NDCGIndex>, boost::noncopyable>("RankNDCGTreePy", py::init<double, double, unsigned, std::size_t, std::size_t, unsigned, bool, double>()));
  expose_Traverser<ErrorTreePy>("ErrorTreeTraverserPy");
  expose_Traverser<RankTreePy<NDCGIndex>>("RankTreeNDCGPy");
}


#endif // DTREEMODULE_CPP
