#ifndef TYPES_HPP
#define TYPES_HPP
#include <map>
#include <vector>
#include <unordered_map>
#include <google/dense_hash_map>

template<typename K, typename V>
using hash_map_t = google::dense_hash_map<K, V>;

using id_type = int64_t;
using group_t = std::vector<id_type>;
using profile_t = hash_map_t<id_type, double>;

#endif // TYPES_HPP
