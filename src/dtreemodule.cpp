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

};

template<typename C, typename X1>
void expose_methods(py::class_<C, X1> c) {
  c.def("init", &C::init_py)
   .def("build", &C::build);
}

BOOST_PYTHON_MODULE(dtreelib)
{
  expose_methods(py::class_<ErrorTreePy, boost::noncopyable>("ErrorTreePy", py::init<double, double, unsigned, std::size_t, std::size_t, unsigned, bool, double>()));
  //expose_methods(py::class_<RankTree<NDCGIndex>>("RankTree", py::init<double, double, unsigned, std::size_t, std::size_t, unsigned, bool, double>()));
}


#endif // DTREEMODULE_CPP
