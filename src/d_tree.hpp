#ifndef D_TREE_HPP
#define D_TREE_HPP

#include "types.hpp"

class DNode{
public:
    DNode() :
        _parent{nullptr}, _children{}, _id{0u}, _splitter_id{0u}, _level{0u}{}
    virtual double prediction(std::size_t _item_id) const = 0;
protected:
    DNode *_parent;
    std::vector<std::unique_ptr<BDNode>> _children;
    std::size_t _id;
    std::size_t _splitter_id;
    unsigned _level;

};

class DTree{
public:
    using ptr_t = DNode*;
    using cptr_t = DNode const*;

public:
    virtual void build() = 0;
    virtual void init(const std::vector<rating_t> &) = 0;
    virtual void init(const std::string &) = 0;
    virtual double splitting_error() const = 0;
    virtual cptr_t traverse(const profile_t &) const = 0;

protected:
    ptr_t *_root;

};

#endif // D_TREE_HPP
