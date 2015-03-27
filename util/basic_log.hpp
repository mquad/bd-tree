#ifndef BASIC_LOG_HPP
#define BASIC_LOG_HPP
#include <iostream>

class BasicLogger{
    std::ostream *out;
public:
    BasicLogger(std::ostream &os){
        out = &os;
    }
    std::ostream &node(const std::size_t node_id, const std::size_t node_level=0u){
        for(std::size_t l{}; l < node_level-1; ++l)
            (*out) << "\t";
        (*out) << "[NODE " << node_id << "]: ";
        return *out;
    }

};

#endif // BASIC_LOG_HPP
