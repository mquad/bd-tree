#ifndef ABD_INDEX_HPP
#define ABD_INDEX_HPP
#include <map>
#include <vector>
#include "stats.hpp"

template<typename Key, typename Stat>
class ABDIndex{
public:
    using score_t = typename Stat::score_t;
    using entry_t = std::vector<score_t>;
protected:
    std::map<Key, entry_t> _index;
public:
    struct bound_t{
        std::size_t _left, _right;

        bound_t(std::size_t left, std::size_t right) : _left{left}, _right{right}{}
        bound_t() : _left{0u}, _right{0u}{}

        std::size_t size() const {return _right - _left;}
        friend bool operator ==(const bound_t &lhs, const bound_t &rhs){
            return (lhs._left == rhs._left) && (rhs._right == rhs._right);
        }
        friend bool operator !=(const bound_t &lhs, const bound_t &rhs){
            return !(lhs == rhs);
        }
        friend std::ostream &operator <<(std::ostream &os, const bound_t &b){
            os << "[" << b._left << ", " << b._right << ")";
            return os;
        }
    };

public:
    ABDIndex() : _index{}{}
    ~ABDIndex(){}

    // accessors for some basic properties
    std::size_t size() const    {return _index.size();}
    entry_t& at(const Key &key)              {return _index.at(key);}
    const entry_t& at(const Key& key) const  {return _index.at(key);}
    entry_t& operator[] (const Key &key)     {return _index[key];}

    decltype(_index.begin()) begin()            {return _index.begin();}
    decltype(_index.end()) end()                {return _index.end();}
    decltype(_index.cbegin()) cbegin() const    {return _index.cbegin();}
    decltype(_index.cend()) cend() const        {return _index.cend();}

    void insert(const Key &key, const score_t &score){
        _index[key].push_back(score);
    }

    // return the -sorted- key vector
    std::vector<Key> keys(){
        return extract_keys(_index);
    }

    void sort_all(){
        for(auto &entry : _index)
            sort_entry(entry.first);
    }

    void sort_entry(const Key &key){
        std::stable_sort(_index.at(key).begin(), _index.at(key).end());
    }

    StatMap<Key, Stat> all_stats() const{
        StatMap<Key, Stat>  stats{};
        for(const auto &entry : _index)
            update_stats(stats, entry.first);
        return stats;
    }

    // updates the stats with all the values associated to a given key
    void update_stats(StatMap<Key, Stat> &stats, const Key key) const{
        if(_index.count(key) > 0){
            for(const auto &score : _index.at(key))
                stats[score._id].update(score);
        }
    }


};

#endif // ABD_INDEX_HPP
