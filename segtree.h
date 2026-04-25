#include <iostream>
#include <cstring>
#include <queue>
#include <cmath>
#include <vector>
#include <stack>
#include "./hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;
#define PUSH_HEAP(vec, ...)      \
  vec.emplace_back(__VA_ARGS__); \
  std::push_heap(vec.begin(), vec.end())

#define POP_HEAP(vec)                    \
  std::pop_heap(vec.begin(), vec.end()); \
  vec.pop_back();

#define TOP_HEAP(vec) vec.front()

template <typename id_t = tableint>
class bitset_t
{
public:
  bitset_t() = delete;
  bitset_t(size_t n) : n_(n)
  {
    size_t n_bytes = (n + 7) / 8;
    size_t aligned_bytes = (n_bytes + 63) & ~63; // Round up to multiple of 64
    data_          = static_cast<uint64_t *>(aligned_alloc(64, aligned_bytes));
    if (data_ == nullptr) {
      throw std::runtime_error("fail to alloc for bitset_t");
    }
  }

  bitset_t(const bitset_t &) = delete;

  bitset_t(bitset_t &&other)
  {
    n_          = other.n_;
    data_       = other.data_;
    other.data_ = nullptr;
  }

  ~bitset_t()
  {
    if (data_)
      free(data_);
  }

  inline __attribute__((always_inline)) void Set(id_t i)  { data_[i / 64] |= 1ULL << (i % 64); }

  inline __attribute__((always_inline)) bool Test(id_t i) const  { return data_[i / 64] & (1ULL << (i % 64)); }

  inline __attribute__((always_inline)) void Reset(id_t i) { data_[i / 64] &= ~(1ULL << (i % 64)); }

  inline __attribute__((always_inline)) auto GetData(id_t i) -> uint64_t * { return &data_[i / 64]; }

  inline __attribute__((always_inline)) void Clear()
  {
    size_t n_bytes = (n_ + 7) / 8;
    memset(data_, 0, n_bytes);
  }

public:
  size_t    n_{};
  uint64_t *data_{nullptr};
};

using dist_t = float;

class SegmentTree {
public:
    int M = 16;                 // Tightly connected with internal dimensionality of the data
    int ef_construction = 256;  // Controls index search speed/build speed tradeoff
    // Initing index
    hnswlib::L2Space *space;
    int dim;
    int max_elements_;
    hnswlib::HierarchicalNSW<float>* alg_hnsw;

    struct Node {
        uint32_t left, right; 
        uint32_t level;
        uint32_t ep_id;
        uint32_t size;
        uint32_t true_elem_in_range_size;
        bool reuse;
    };

    Node *tree;
    uint32_t treeNodeSize;
    uint32_t cnt;
    uint32_t l, r;
    uint32_t root;
    char segTreeMaxLevel = 1;
    uint32_t* attrs_hnsw;
    bitset_t<uint32_t> *visit_set;
    float rebuildRatio = 1.5;

    uint32_t metric_dist_comps_{0};
    void init(int dim_ = 1, int max_elements = 1000000, int M = 16,int ef_construction = 256,  float rebuildRatio_ = 1.5){
        segTreeMaxLevel = 0;
        l=0;
        r=pow(2,segTreeMaxLevel)-1;
        cnt=0;
        root=0;
        treeNodeSize = 100;
        tree = (Node *)malloc(treeNodeSize * sizeof(Node));
        memset(tree, 0, treeNodeSize * sizeof(Node));

        space = new hnswlib::L2Space(dim_);
        dim = dim_;
        max_elements_ = max_elements;
        attrs_hnsw = (uint32_t *)malloc(max_elements_ * sizeof(uint32_t));
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, max_elements_,attrs_hnsw, M, ef_construction,segTreeMaxLevel);
        visit_set =  new bitset_t<uint32_t>(max_elements_);
        rebuildRatio = rebuildRatio_;
    }

    ~SegmentTree(){
        free(tree);
        tree = nullptr;
        delete space;
        delete alg_hnsw;
        space = nullptr;
        alg_hnsw = nullptr;
        free(attrs_hnsw);
        delete visit_set;
    }
    struct CompareByFirst1 {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    void insertOne(uint32_t attr, void *element,uint64_t label){
        if(attr<l){
           //cout<<"attr can not be less than "<<l<<endl;
            return;
        }
        while(attr>r){
            segTreeMaxLevel+=1;
            r=pow(2,segTreeMaxLevel)-1;
            uint32_t temp_root = root;
            ++cnt;
            root = cnt;
            tree[root].level = segTreeMaxLevel;
            if(temp_root!=0 && tree[temp_root].size!=0)
                tree[root].left = temp_root,tree[root].size=tree[temp_root].size;
            else
                tree[root].left = 0;
            tree[root].right = 0;
            tree[root].ep_id = tree[tree[root].left].ep_id;
            tree[root].reuse = true;

            alg_hnsw->segtree_inc();
            if(tree[root].left!=0&& tree[root].size!=0)
            {

                for(uint32_t i =0;i < alg_hnsw->cur_element_count;i++){
                    alg_hnsw->modify_LinksLevel(i,tree[root].level,0,tree[tree[root].left].level,0);
                }
                tree[root].ep_id = tree[tree[root].left].ep_id;
            }
            if(cnt >= treeNodeSize-segTreeMaxLevel-2){
                tree = (Node *)realloc(tree, 2* treeNodeSize * sizeof(Node));
                if(tree == nullptr)
                    throw std::runtime_error("realloc failed");
                memset(tree+ treeNodeSize, 0, treeNodeSize * sizeof(Node));;
                treeNodeSize *= 2;
            }
        }

        if(cnt >= treeNodeSize-segTreeMaxLevel-2){
            tree = (Node *)realloc(tree, 2* treeNodeSize * sizeof(Node));
            if(tree == nullptr)
                throw std::runtime_error("realloc failed");
            memset(tree+ treeNodeSize, 0, treeNodeSize * sizeof(Node));;
            treeNodeSize *= 2;
        }

        vector<vector<uint32_t>> actions;
        insertOne_actions(segTreeMaxLevel,0,root,l,r,attr, element,label,actions);

        int size = actions.size();
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, hnswlib::HierarchicalNSW<float>::CompareByFirst> top_candidates;
        int modify=0;

        for(int i=size-1;i>=0;i--){
            if(actions[i][0]==0){
                if(modify==0){
                    tree[actions[i][1]].ep_id = alg_hnsw->addPointInSegtreewithCands(tree[actions[i][1]].ep_id,tree[actions[i][1]].level,actions[i][2],element,label,attr,actions[i][3],actions[i][4],top_candidates);
                }else{
                    alg_hnsw->modify_LinksLevel(alg_hnsw->getInnerLabel(label),actions[i+1][5],actions[i+1][6],tree[actions[i+1][1]].level,actions[i+1][2]);
                    tree[actions[i][1]].ep_id = tree[actions[i+1][1]].ep_id;
                    modify=0;
                }
            }else{
                if(modify==0){
                    tree[actions[i][1]].ep_id = alg_hnsw->addPointInSegtreewithCands(tree[actions[i][1]].ep_id,tree[actions[i][1]].level,actions[i][2],element,label,attr,actions[i][3],actions[i][4],top_candidates);
                }else{
                    alg_hnsw->modify_LinksLevel(alg_hnsw->getInnerLabel(label),actions[i+1][5],actions[i+1][6],tree[actions[i+1][1]].level,actions[i+1][2]);
                    tree[actions[i][1]].ep_id = tree[actions[i+1][1]].ep_id;
                }
                modify++;
            }
        }
    }

    void updateOneLabel(uint32_t attr, uint64_t label){
        auto id = alg_hnsw->getInnerLabel2(label);
        if(id == max_elements_){
            return;
        }
        if(attr<l){
           //cout<<"attr can not be less than "<<l<<endl;
            return;
        }
        while(attr>r){
            segTreeMaxLevel+=1;
            r=pow(2,segTreeMaxLevel)-1;
            uint32_t temp_root = root;
            ++cnt;
            root = cnt;
            tree[root].level = segTreeMaxLevel;
            if(temp_root!=0 && tree[temp_root].size!=0)
                tree[root].left = temp_root,tree[root].size=tree[temp_root].size;
            else
                tree[root].left = 0;
            tree[root].right = 0;
            tree[root].ep_id = tree[tree[root].left].ep_id;
            tree[root].reuse = true;

            alg_hnsw->segtree_inc();
            if(tree[root].left!=0&& tree[root].size!=0)
            {
                for(uint32_t i =0;i < alg_hnsw->cur_element_count;i++){
                    alg_hnsw->modify_LinksLevel(i,tree[root].level,0,tree[tree[root].left].level,0);
                }
                tree[root].ep_id = tree[tree[root].left].ep_id;
            }
            if(cnt >= treeNodeSize-segTreeMaxLevel-1){
                tree = (Node *)realloc(tree, 2* treeNodeSize * sizeof(Node));
                if(tree == nullptr)
                    throw std::runtime_error("realloc failed");
                memset(tree+ treeNodeSize, 0, treeNodeSize * sizeof(Node));;
                treeNodeSize *= 2;
            }
        }
        if(cnt >= treeNodeSize-segTreeMaxLevel-1){
            tree = (Node *)realloc(tree, 2* treeNodeSize * sizeof(Node));
            if(tree == nullptr)
                throw std::runtime_error("realloc failed");
            memset(tree+ treeNodeSize, 0, treeNodeSize * sizeof(Node));;
            treeNodeSize *= 2;
        }

        vector<vector<uint32_t>> actions;
        void *element = (void *)(alg_hnsw->getDataByInternalId(alg_hnsw->getInnerLabel(label)));
        updateOnelabel_actions(segTreeMaxLevel,0,root,l,r,attr, element,label,actions,attrs_hnsw[alg_hnsw->getInnerLabel(label)]);
        int size = actions.size();
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, hnswlib::HierarchicalNSW<float>::CompareByFirst> top_candidates;
        int modify=0;
        
        for(int i=size-1;i>=0;i--){
            if(actions[i][0]==0){
                if(modify==0){
                    tree[actions[i][1]].ep_id = alg_hnsw->updatePointLabelInSegtreewithCands(tree[actions[i][1]].ep_id,tree[actions[i][1]].level,actions[i][2],element,label,attr,actions[i][3],actions[i][4],top_candidates);
                }else{
                    alg_hnsw->modify_LinksLevel(alg_hnsw->getInnerLabel(label),actions[i+1][5],actions[i+1][6],tree[actions[i+1][1]].level,actions[i+1][2]);
                    tree[actions[i][1]].ep_id = tree[actions[i+1][1]].ep_id;
                    modify=0;
                }
            }else{
                if(modify==0){
                    tree[actions[i][1]].ep_id = alg_hnsw->updatePointLabelInSegtreewithCands(tree[actions[i][1]].ep_id,tree[actions[i][1]].level,actions[i][2],element,label,attr,actions[i][3],actions[i][4],top_candidates);
                }else{
                    alg_hnsw->modify_LinksLevel(alg_hnsw->getInnerLabel(label),actions[i+1][5],actions[i+1][6],tree[actions[i+1][1]].level,actions[i+1][2]);
                    tree[actions[i][1]].ep_id = tree[actions[i+1][1]].ep_id;
                    
                }
                modify++;
            }
        }
    }

    
    void insertOne_actions(uint32_t level, uint32_t x ,uint32_t& root, uint32_t l, uint32_t r, uint32_t attr, void *element,uint64_t label,vector<vector<uint32_t>> &actions){
        if (root == 0) {
            ++cnt;
            root = cnt;
            tree[root].level = level;
        }
    
        if(l == r){
            tree[root].size++; 
            tree[root].true_elem_in_range_size ++;
            if(tree[root].size==1){
                tree[root].ep_id = -1;
            }
            actions.push_back({0,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});
            return;
        }
        uint32_t mid = (l + r) >> 1;

        if (attr <= mid) {
            insertOne_actions(tree[root].level-1,(x<<1),tree[root].left, l, mid, attr, element, label,actions);
        }else{
            insertOne_actions(tree[root].level-1,(x<<1)+1,tree[root].right, mid+1, r, attr, element, label,actions);
        }
        tree[root].size ++;
        tree[root].true_elem_in_range_size ++;


        if(tree[root].left == 0 && tree[root].right != 0){
            uint32_t l1 = 1;
            uint32_t l2 = 1;
            if(tree[root].size > 1){
                auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[root].ep_id,tree[root].level);
                uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                if(level==nullptr)
                    l1 = -1;
                else
                    l1 = *level;
                auto levels2 = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[tree[root].right].ep_id,tree[tree[root].right].level);
                uint32_t* level2 = alg_hnsw->get_LinksLevel(levels2,(x<<1)+1);
                l2 = *level2;
            }
            if(l1!=l2)
            {
                queue<uint32_t> candidate_set;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;

                candidate_set.emplace(tree[tree[root].right].ep_id);
                visited_array[tree[tree[root].right].ep_id] = visited_array_tag;
                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
                    alg_hnsw->modify_LinksLevel(cur_id,tree[root].level,x,tree[tree[root].right].level,(x<<1)+1);
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[root].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                    if(level==nullptr)
                        continue;

                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif                    
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(datal + j+1)), _MM_HINT_T0);
#endif 
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                }

                visited_list_pool_->releaseVisitedList(vl);
                actions.push_back({1,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});
            }else{
                actions.push_back({1,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});
            }
            tree[root].ep_id = tree[tree[root].right].ep_id;
            tree[root].reuse = true;
        }else if(tree[root].right == 0 && tree[root].left !=0){
            uint32_t l1 = 1;
            uint32_t l2 = 1;
            if(tree[root].size > 1){
                auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[root].ep_id,tree[root].level);
                uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                if(level==nullptr)
                    l1 = -1;
                else
                    l1 = *level;
                auto levels2 = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[tree[root].left].ep_id,tree[tree[root].left].level);
                uint32_t* level2 = alg_hnsw->get_LinksLevel(levels2,(x<<1));
                l2 = *level2;

            }
            if(l1!=l2)
            {
                queue<uint32_t> candidate_set;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;

                candidate_set.emplace(tree[tree[root].left].ep_id);
                visited_array[tree[tree[root].left].ep_id] = visited_array_tag;
                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
                    alg_hnsw->modify_LinksLevel(cur_id,tree[root].level,x,tree[tree[root].left].level,(x<<1));
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[root].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                    if(level==nullptr)
                        continue;
                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif                    
                    tableint *datal = (tableint *) (data + 1);

                    for (size_t j = 0; j < size; j++) {
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(datal + j+1)), _MM_HINT_T0);
#endif 
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                }

                visited_list_pool_->releaseVisitedList(vl);
                actions.push_back({1,root,x,l,r,tree[tree[root].left].level,(x<<1)});
            }else{
                actions.push_back({1,root,x,l,r,tree[tree[root].left].level,(x<<1)});
            }
            tree[root].ep_id = tree[tree[root].left].ep_id;
            tree[root].reuse = true;
        }
        else
        {
            if(tree[root].reuse == true){

                int low=0;
                uint32_t ll,rr,xx;
                if(tree[tree[root].right].ep_id==-1){
                    low = tree[root].left,xx=(x<<1);
                }else
                    low = tree[root].right,xx=(x<<1)+1;
                while(tree[low].level>0){
                    if(tree[low].right==0){
                        low = tree[low].left,xx=(xx<<1);
                    }else if(tree[low].left==0)
                        low = tree[low].right,xx=(xx<<1)+1;
                    else
                        break;
                }
                
                queue<uint32_t> candidate_set;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;
                candidate_set.emplace(tree[low].ep_id);
                visited_array[tree[low].ep_id] = visited_array_tag;
                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
                    alg_hnsw->physicalReuse(cur_id,tree[root].level,x,tree[low].level,xx);
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[root].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                    if(level==nullptr)
                        continue;

                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif          
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(datal + j+1)), _MM_HINT_T0);
#endif 
                        hnswlib::tableint candidate_id = *(datal + j);

                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                }

                visited_list_pool_->releaseVisitedList(vl);


                tree[root].ep_id = tree[low].ep_id;
                tree[root].reuse = false;
            }
            actions.push_back({0,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});
        }
    }

    void logicDelete(uint32_t level, uint32_t x ,uint32_t& root, uint32_t l, uint32_t r,uint64_t label,uint32_t ori_attr){
        if(l == r){
            if(tree[root].size==0){
                tree[root].ep_id = -1;
            }
            bool flag = false;
            flag = alg_hnsw->isInsubIndex(tree[root].ep_id,level,x,nullptr,label);
            if(flag){
                tree[root].true_elem_in_range_size--;
            }
            return;
        }
        bool flag = false;
        flag = alg_hnsw->isInsubIndex(tree[root].ep_id,level,x,nullptr,label);
        if(flag){
            tree[root].true_elem_in_range_size--;
        }
        uint32_t mid = (l + r) >> 1;

        if (ori_attr <= mid) {
            logicDelete(tree[root].level-1,(x<<1),tree[root].left, l, mid,label, ori_attr);
        }else{
            logicDelete(tree[root].level-1,(x<<1)+1,tree[root].right, mid+1, r, label, ori_attr);
        }
    }

    void updateOnelabel_actions(uint32_t level, uint32_t x ,uint32_t& root, uint32_t l, uint32_t r, uint32_t attr, void *element,uint64_t label,vector<vector<uint32_t>> &actions,uint32_t ori_attr){
        if (root == 0) {
            ++cnt;
            root = cnt;
            tree[root].level = level;
        }
    
        if(l == r){
            if(tree[root].size==0){
                tree[root].ep_id = -1;
            }
            
            bool flag = false;
            flag = alg_hnsw->isInsubIndex(tree[root].ep_id,level,x,element,label);
            if(!flag){
                tree[root].size++;
                tree[root].true_elem_in_range_size ++;
            }

            attrs_hnsw[alg_hnsw->getInnerLabel(label)]=attr;
            actions.push_back({0,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});

            if((1.0*tree[root].size>rebuildRatio*tree[root].true_elem_in_range_size) &&(tree[root].true_elem_in_range_size>1)&&(tree[root].size>M)){
                tree[root].size = 0;
                tree[root].true_elem_in_range_size = 0;
                queue<uint32_t> candidate_set;
                vector<uint32_t> true_elem_in_range;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;

                candidate_set.emplace(tree[root].ep_id);
                visited_array[tree[root].ep_id] = visited_array_tag;
                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
    
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[root].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                    if(level==nullptr)
                        continue;
                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif          
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                    if(attrs_hnsw[cur_id]>=l&&attrs_hnsw[cur_id]<=r){
                        true_elem_in_range.emplace_back(cur_id);
                        alg_hnsw->rebuildDeleteEdges(cur_id,tree[root].level,x,false);
                    }else{
                        alg_hnsw->rebuildDeleteEdges(cur_id,tree[root].level,x,true);
                    }
                }

                tree[root].ep_id = -1;
                int cnt = true_elem_in_range.size();
                for(int i=0;i<true_elem_in_range.size();i++){
                    uint32_t cur_id = true_elem_in_range[i];
                    tree[root].ep_id = alg_hnsw->rebuildAddPointInSegtree(tree[root].ep_id,tree[root].level,x,alg_hnsw->getDataByInternalId(cur_id),alg_hnsw->getExternalLabel(cur_id));
                }
                tree[root].size += cnt;
                tree[root].true_elem_in_range_size += cnt;
                visited_list_pool_->releaseVisitedList(vl);
            }
            return;
        }
        uint32_t mid = (l + r) >> 1;

        if (attr <= mid) {
            if(ori_attr > mid && ori_attr<=r){
                logicDelete(tree[root].level-1,(x<<1)+1,tree[root].right, mid+1, r,label,ori_attr);
            }
            updateOnelabel_actions(tree[root].level-1,(x<<1),tree[root].left, l, mid, attr, element, label,actions,ori_attr);
        }else{
            if(ori_attr <= mid && ori_attr>=l){
                logicDelete(tree[root].level-1,(x<<1),tree[root].left, l, mid,label,ori_attr);
            }
            updateOnelabel_actions(tree[root].level-1,(x<<1)+1,tree[root].right, mid+1, r, attr, element, label,actions,ori_attr);
        }

        if(!(alg_hnsw->isInsubIndex(tree[root].ep_id,tree[root].level,x,element,label))){
            tree[root].size ++;
            tree[root].true_elem_in_range_size ++;
        }
        if(tree[root].left == 0 && tree[root].right != 0){
            uint32_t l1 = 1;
            uint32_t l2 = 1;
            if(tree[root].size > 1){
                auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[root].ep_id,tree[root].level);
                uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                if(level==nullptr)
                    l1 = -1;
                else
                    l1 = *level;
                auto levels2 = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[tree[root].right].ep_id,tree[tree[root].right].level);
                uint32_t* level2 = alg_hnsw->get_LinksLevel(levels2,(x<<1)+1);
                l2 = *level2;
            }
            if(l1!=l2)
            {
                queue<uint32_t> candidate_set;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;

                candidate_set.emplace(tree[tree[root].right].ep_id);
                visited_array[tree[tree[root].right].ep_id] = visited_array_tag;
                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
                    alg_hnsw->modify_LinksLevel(cur_id,tree[root].level,x,tree[tree[root].right].level,(x<<1)+1);
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[tree[root].right].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,(x<<1)+1);
                    if(level==nullptr)
                        continue;
                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif          
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                }

                visited_list_pool_->releaseVisitedList(vl);
                actions.push_back({1,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});
            }else{
                actions.push_back({1,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});
            }
            tree[root].ep_id = tree[tree[root].right].ep_id;
            tree[root].reuse = true;
        }else if(tree[root].right == 0 && tree[root].left !=0){
            uint32_t l1 = 1;
            uint32_t l2 = 1;
            if(tree[root].size > 1){
                auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[root].ep_id,tree[root].level);
                uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                if(level==nullptr)
                    l1 = -1;
                else
                    l1 = *level;
                auto levels2 = alg_hnsw->get_segTreelevel_2_LinksLevel(tree[tree[root].left].ep_id,tree[tree[root].left].level);
                uint32_t* level2 = alg_hnsw->get_LinksLevel(levels2,(x<<1));
                l2 = *level2;
            }
            if(l1!=l2)
            {
                queue<uint32_t> candidate_set;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;

                candidate_set.emplace(tree[tree[root].left].ep_id);
                visited_array[tree[tree[root].left].ep_id] = visited_array_tag;
                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
                    alg_hnsw->modify_LinksLevel(cur_id,tree[root].level,x,tree[tree[root].left].level,(x<<1));
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[tree[root].left].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,(x<<1));
                    if(level==nullptr)
                        continue;
                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif          
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                }

                visited_list_pool_->releaseVisitedList(vl);
                actions.push_back({1,root,x,l,r,tree[tree[root].left].level,(x<<1)});

            }else{
                actions.push_back({1,root,x,l,r,tree[tree[root].left].level,(x<<1)});
            }
            tree[root].ep_id = tree[tree[root].left].ep_id;
            tree[root].reuse = true;
        }else{

            if( tree[root].reuse == true){

                int low=0;
                uint32_t ll,rr,xx;
                if(tree[tree[root].right].ep_id==-1){
                    low = tree[root].left,xx=(x<<1);
                }else
                    low = tree[root].right,xx=(x<<1)+1;
                while(tree[low].level>0){
                    if(tree[low].right==0){
                        low = tree[low].left,xx=(xx<<1);
                    }else if(tree[low].left==0)
                        low = tree[low].right,xx=(xx<<1)+1;
                    else
                        break;
                }
                
                queue<uint32_t> candidate_set;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;

                candidate_set.emplace(tree[low].ep_id);
                visited_array[tree[low].ep_id] = visited_array_tag;
                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
                    alg_hnsw->physicalReuse(cur_id,tree[root].level,x,tree[low].level,xx);
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[low].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,xx);
                    if(level==nullptr)
                        continue;
                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif          
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                }

                visited_list_pool_->releaseVisitedList(vl);

                tree[root].ep_id = tree[low].ep_id;
                tree[root].reuse = false;
            }
            actions.push_back({0,root,x,l,r,tree[tree[root].right].level,(x<<1)+1});
        }

        if(!(alg_hnsw->isInsubIndex(tree[root].ep_id,tree[root].level,x,element,label))){


            if((1.0*tree[root].size>rebuildRatio*tree[root].true_elem_in_range_size) &&(tree[root].true_elem_in_range_size>1)&&(tree[root].size>M)){
                int low=0;
                uint32_t xx;
                if(tree[root].right == 0 && tree[root].left !=0){
                    tree[root].ep_id = tree[tree[root].left].ep_id;
                    tree[root].size = tree[tree[root].left].size;
                    tree[root].true_elem_in_range_size = tree[tree[root].left].true_elem_in_range_size;
                    return;
                }else if(tree[root].right != 0 && tree[root].left ==0){
                    tree[root].ep_id = tree[tree[root].right].ep_id;
                    tree[root].size = tree[tree[root].right].size;
                    tree[root].true_elem_in_range_size = tree[tree[root].right].true_elem_in_range_size;
                    return;
                }else{
                    if(tree[root].reuse == true){
                        if(tree[tree[root].right].ep_id==-1){
                            low = tree[root].left,xx=(x<<1);
                        }else
                            low = tree[root].right,xx=(x<<1)+1;
                        while(tree[low].level>0){
                            if(tree[low].right==0){
                                low = tree[low].left,xx=(xx<<1);
                            }else if(tree[low].left==0)
                                low = tree[low].right,xx=(xx<<1)+1;
                            else
                                break;
                        }
                    }else{
                        if(tree[tree[root].right].size<tree[tree[root].left].size){
                            low = tree[root].left,xx=(x<<1);
                        }else
                            low = tree[root].right,xx=(x<<1)+1;
                    }
                }
                tree[root].size = 0;
                tree[root].true_elem_in_range_size = 0;
                queue<uint32_t> candidate_set;
                vector<uint32_t> true_elem_in_range;
                
                auto visited_list_pool_ = std::unique_ptr<hnswlib::VisitedListPool>(new hnswlib::VisitedListPool(1, max_elements_));
                auto *vl = visited_list_pool_->getFreeVisitedList();
                auto *visited_array = vl->mass;
                auto visited_array_tag = vl->curV;

                candidate_set.emplace(tree[root].ep_id);
                visited_array[tree[root].ep_id] = visited_array_tag;


                while(!candidate_set.empty()){
                    uint32_t cur_id = candidate_set.front();
                    candidate_set.pop();
    
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[root].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                    if(level==nullptr)
                        continue;
                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif          
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set.emplace(candidate_id);
                    }
                    if(attrs_hnsw[cur_id]>=l&&attrs_hnsw[cur_id]<=r){
                        true_elem_in_range.emplace_back(cur_id);
                        alg_hnsw->rebuildDeleteEdges(cur_id,tree[root].level,x,false);
                    }else{
                        alg_hnsw->rebuildDeleteEdges(cur_id,tree[root].level,x,true);
                    }
                }
            
                queue<uint32_t> candidate_set2;
                vl->curV++;
                visited_array_tag = vl->curV;

                candidate_set2.emplace(tree[low].ep_id);
                visited_array[tree[low].ep_id] = visited_array_tag;
                int fuyong_size=0;
                while(!candidate_set2.empty()){
                    uint32_t cur_id = candidate_set2.front();
                    tree[root].size++;
                    fuyong_size++;
                    if(attrs_hnsw[cur_id]>=l&&attrs_hnsw[cur_id]<=r){
                        tree[root].true_elem_in_range_size++;
                    }
                    candidate_set2.pop();
                    alg_hnsw->physicalReuse4rebuild(cur_id,tree[root].level,x,tree[low].level,xx);
                    int *data; 
                    auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[low].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,xx);
                    if(level==nullptr)
                        continue;
                    data = (int*)alg_hnsw->get_linklist(cur_id, *level);

                    size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
#endif          
                    tableint *datal = (tableint *) (data + 1);
                    for (size_t j = 0; j < size; j++) {
                        hnswlib::tableint candidate_id = *(datal + j);
                        if (visited_array[candidate_id] == visited_array_tag) continue;
                        visited_array[candidate_id] = visited_array_tag;
                        candidate_set2.emplace(candidate_id);
                    }
                }


                tree[root].ep_id = tree[low].ep_id;

                int cnt = true_elem_in_range.size();
                for(int i=0;i<true_elem_in_range.size();i++){
                    uint32_t cur_id = true_elem_in_range[i];

                    uint32_t* levels = alg_hnsw->get_segTreelevel_2_LinksLevel(cur_id,tree[root].level);
                    uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                    if(level==nullptr)
                        continue;
                    linklistsizeint *ll_another = alg_hnsw->get_linklist(cur_id, *level);
                    size_t size = alg_hnsw->getListCount((linklistsizeint*)ll_another);

                    if(size==0&&(fuyong_size>1))
                        tree[root].ep_id = alg_hnsw->rebuildAddPointInSegtree(tree[root].ep_id,tree[root].level,x,alg_hnsw->getDataByInternalId(cur_id),alg_hnsw->getExternalLabel(cur_id));
                    else
                        cnt--;
                }
                tree[root].size += cnt;
                tree[root].true_elem_in_range_size += cnt;
                visited_list_pool_->releaseVisitedList(vl);
            }
        }

    }

    
    //返回包含于attr_l~attr_r的树节点索引以及其最近祖先
    void query(vector<uint32_t> &ret, uint32_t &maxNode, uint32_t attr_l, uint32_t attr_r){ 
        query(ret,maxNode,root,l,r,attr_l,attr_r);
    }
    void query(vector<uint32_t> &ret, uint32_t &maxNode,uint32_t root, uint32_t l, uint32_t r, uint32_t &attr_l, uint32_t &attr_r){ 
        
        if(attr_l <= l &&  r <= attr_r){
            ret.emplace_back(root);
        }
        else{
            uint32_t mid = (l + r) >> 1;
            if(maxNode == 0 && l <= attr_l && attr_l <= mid && mid < attr_r && attr_r <= r){
                maxNode = root;
            }
            if (tree[root].left!=0 && attr_l <= mid) {
                query(ret, maxNode,tree[root].left, l, mid, attr_l, attr_r);
            }
            if (tree[root].right!=0 && attr_r >= mid+1) {
                query(ret, maxNode,tree[root].right, mid+1, r, attr_l, attr_r);
            }
        }
    }

    struct seglevel_and_range_ {
        uint32_t root;
        uint32_t seglevel;
        uint32_t x;
        uint32_t left;
        uint32_t right;
        seglevel_and_range_(uint32_t root,uint32_t level, uint32_t x,uint32_t l,uint32_t r) : root(root),seglevel(level),x(x), left(l),right(r) {}
    };

    void query(vector<seglevel_and_range_> &ret, uint32_t level,uint32_t x,uint32_t &maxNode,uint32_t root, uint32_t l, uint32_t r, uint32_t &attr_l, uint32_t &attr_r){ 
        
        if(attr_l <= l &&  r <= attr_r){
            ret.emplace_back(root,level,x,l,r);
        }
        else{
            uint32_t mid = (l + r) >> 1;
            if(maxNode == 0 && l <= attr_l && attr_l <= mid && mid < attr_r && attr_r <= r){
                maxNode = root;
                
            }
            if (tree[root].left!=0 && attr_l <= mid) {
                query(ret, level-1,x<<1,maxNode,tree[root].left, l, mid, attr_l, attr_r);
            }
            if (tree[root].right!=0 && attr_r >= mid+1) {
                query(ret, level-1,(x<<1)+1, maxNode,tree[root].right, mid+1, r, attr_l, attr_r);
            }
        }
    }

    void query(vector<seglevel_and_range_> &ret, uint32_t level,uint32_t x,uint32_t &maxNode,uint32_t &maxNode_x,uint32_t root, uint32_t l, uint32_t r, uint32_t &attr_l, uint32_t &attr_r){ 
        
        if(attr_l <= l &&  r <= attr_r){
            ret.emplace_back(root,level,x,l,r);
        }
        else{
            uint32_t mid = (l + r) >> 1;
            if(maxNode == 0 && l <= attr_l && attr_l <= mid && mid < attr_r && attr_r <= r){
                maxNode = root;
                maxNode_x  =x;
            }
            if (tree[root].left!=0 && attr_l <= mid) {
                query(ret, level-1,x<<1,maxNode,maxNode_x,tree[root].left, l, mid, attr_l, attr_r);
            }
            if (tree[root].right!=0 && attr_r >= mid+1) {
                query(ret, level-1,(x<<1)+1, maxNode,maxNode_x,tree[root].right, mid+1, r, attr_l, attr_r);
            }
        }
    }

    void query4ep(vector<seglevel_and_range_> &ret, uint32_t level, uint32_t x,
            uint32_t &maxNode, uint32_t &maxNode_x,
            uint32_t root, uint32_t l, uint32_t r,
            uint32_t &attr_l, uint32_t &attr_r) {

        if (!ret.empty()) return;

        if (r < attr_l || l > attr_r) {
            return;
        }

        if (l == r) {
            if (root != 0) {
                ret.emplace_back(root, level, x, l, r);
            }
            return;
        }

        uint32_t mid = (l + r) >> 1;

        if(maxNode == 0 && l <= attr_l && attr_l <= mid && mid < attr_r && attr_r <= r){
            maxNode = root;
            maxNode_x  =x;
        }

        if (tree[root].left != 0) {
            query(ret, level - 1, x << 1,
                maxNode, maxNode_x,
                tree[root].left, l, mid,
                attr_l, attr_r);
        }

        if (ret.empty() && tree[root].right != 0) {
            query(ret, level - 1, (x << 1) + 1,
                maxNode, maxNode_x,
                tree[root].right, mid + 1, r,
                attr_l, attr_r);
        }
    }


    void printQueryResult(vector<uint32_t> queryResult){
        cout << "query results:" << endl;
        for (auto node : queryResult) {
            cout << "node[" << node << "] , ( level=" << tree[node].level << " size=" << tree[node].size << " ) ";
        }
        cout << endl;
    }
    
    void printTreeLevelOrder(){
        printTreeLevelOrder(root,l,r);
    }
    struct QueueNode {
        uint32_t node;  
        uint32_t l;  
        uint32_t r;  
        uint64_t size;
        uint32_t depth;  
        
        QueueNode(uint32_t n, uint32_t ll, uint32_t rr, uint64_t s, uint32_t d) : node(n), l(ll), r(rr), size(s), depth(d) {}
    };
    
    void printTreeLevelOrder(int root, int rootL, int rootR) {
        if (root == 0) {
            cout << "empty segtree" << endl;
            return;
        }

        queue<QueueNode> q;
        q.push(QueueNode(root, rootL, rootR,tree[root].size,0)); 
        int currentDepth = 0;

        cout << "level travel" << endl;
        while (!q.empty()) {
            QueueNode curr = q.front();
            q.pop();

            if (curr.depth != currentDepth) {
                currentDepth = curr.depth;
                cout << endl; 
                cout << endl; 
                cout << "currentDepth: " << currentDepth << endl;
            }

            if(currentDepth>5){
                break;
            }

            cout << "node[" << curr.node << "](" << curr.l << "," << curr.r << ", size = "<<curr.size<< " ) ";

            if (tree[curr.node].left != 0) {
                int mid = (curr.l + curr.r) >> 1;
                q.push(QueueNode(tree[curr.node].left, curr.l, mid, tree[tree[curr.node].left].size,curr.depth + 1));
            }

            if (tree[curr.node].right != 0) {
                int mid = (curr.l + curr.r) >> 1;
                q.push(QueueNode(tree[curr.node].right, mid + 1, curr.r, tree[tree[curr.node].right].size, curr.depth + 1));
            }
        }
        cout << endl;
    }


    struct candidate_set_t {
        dist_t dist;
        tableint id;
        uint32_t seglevel;
        uint32_t x;
        candidate_set_t( dist_t dist, tableint id,uint32_t level, uint32_t x) : id(id), dist(dist), seglevel(level),x(x){}
        bool operator<(const candidate_set_t &rhs) const { return dist > rhs.dist; }
    };
    struct CompareByFirst {
        constexpr bool operator()(candidate_set_t const& a,
            candidate_set_t const& b) const noexcept {
            return a.dist < b.dist;
        }
    };
    struct top_candidate_set_t {
        dist_t dist;
        tableint id;
        top_candidate_set_t( dist_t dist, tableint id) : id(id), dist(dist) {}
        bool operator<(const top_candidate_set_t &rhs) const { return dist < rhs.dist; }
    };
    struct CompareByFirst2 {
        constexpr bool operator()(top_candidate_set_t const& a,
            top_candidate_set_t const& b) const noexcept {
            return a.dist < b.dist;
        }
    };



std::vector<uint32_t> get_nearest_neighbors(void *data_point, uint32_t attr_l, uint32_t attr_r, uint32_t k, uint32_t top_k) { 
        std::vector<seglevel_and_range_> seglevel_and_range;
        uint32_t max_size_node = 0;
        query(seglevel_and_range,segTreeMaxLevel,0,max_size_node,root,l,r,attr_l,attr_r);

        
        uint32_t max_size_node_level = tree[max_size_node].level;
        if(max_size_node==0)
            max_size_node_level = seglevel_and_range[0].seglevel;


        std::vector<top_candidate_set_t> top_candidates;
        top_candidates.reserve(k+1);
        std::vector<candidate_set_t> candidate_set;
        candidate_set.reserve(k<<2);

        std::vector<seglevel_and_range_> seglevel_and_range2;
        uint32_t max_size_node2= 0;
        uint32_t max_size_node2_x= 0;
        query(seglevel_and_range2,seglevel_and_range[0].seglevel,seglevel_and_range[0].x, max_size_node2,max_size_node2_x,seglevel_and_range[0].root,seglevel_and_range[0].left,seglevel_and_range[0].right,attr_l,attr_l);
        tableint ep_id;
        uint32_t x;
        if(seglevel_and_range2.size()>0){
            ep_id = tree[seglevel_and_range2[0].root].ep_id;
            x = seglevel_and_range2[0].x;
        }else{
            ep_id = tree[max_size_node2].ep_id;
            x = max_size_node2_x;
        }


        dist_t lowerBound = alg_hnsw->fstdistfunc_(data_point, alg_hnsw->getDataByInternalId(ep_id), alg_hnsw->dist_func_param_);
        //metric_dist_comps_++;

        for (uint32_t level = 0; level < max_size_node_level; level++) {
            bool changed = true;
            while (changed) {
                changed = false;

                auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(ep_id,level);
                uint32_t* level1 = alg_hnsw->get_LinksLevel(levels,x);
                if(level1==nullptr)
                    break;
                int *data = (int*)alg_hnsw->get_linklist(ep_id, *level1);

                size_t size = alg_hnsw->getListCount((linklistsizeint*)data);

#ifdef USE_SSE
        _mm_prefetch(alg_hnsw->getDataByInternalId(data[1]), _MM_HINT_T0);
        _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
        _mm_prefetch((char *) (data + 1 ) + 64, _MM_HINT_T0);
#endif
                tableint *datal = (tableint *) (data + 1);

                for (size_t i = 0; i < size; i++) {
                    
                    tableint cand = datal[i];

                        uint32_t attr = attrs_hnsw[cand];
                        if(attr < attr_l){
                            continue;
                        }else if(attr > attr_r){
                            break;
                        }
                    
#ifdef USE_SSE
        _mm_prefetch(alg_hnsw->getDataByInternalId(datal[i+1]), _MM_HINT_T0);
#endif
                    
                    dist_t d = alg_hnsw->fstdistfunc_(data_point, alg_hnsw->getDataByInternalId(cand), alg_hnsw->dist_func_param_);
                    //metric_dist_comps_++;
                    if (d < lowerBound) {
                        lowerBound = d;
                        ep_id = cand;
                        //changed = true;
                        break;
                    }
                }
            }
            x/=2;
        }
        
        auto attr = attrs_hnsw[ep_id];
        for(auto &n : seglevel_and_range){
            if(n.left <= attr && n.right >= attr){
                PUSH_HEAP(top_candidates, lowerBound, ep_id);
                PUSH_HEAP(candidate_set, lowerBound, ep_id,n.seglevel,n.x);
                visit_set->Set(ep_id);
                break;
            }
        }

        lowerBound = TOP_HEAP(top_candidates).dist;

        while (!candidate_set.empty()) {
            candidate_set_t current_node_pair =  TOP_HEAP(candidate_set);

#ifdef USE_SSE
            _mm_prefetch(alg_hnsw->get_segTreelevel_2_LinksLevel(current_node_pair.id,0),  ///////////
                                         _MM_HINT_T0);  ////////////////////////
#endif
            dist_t candidate_dist = current_node_pair.dist;
            tableint current_node_id = current_node_pair.id;
            uint32_t current_node_seglevel = current_node_pair.seglevel;
            uint32_t x = current_node_pair.x;
            uint32_t current_node_x = current_node_pair.x;

            if (candidate_dist > lowerBound) {
                break;
            }
            POP_HEAP(candidate_set);
            for(uint32_t i = current_node_seglevel; i <= max_size_node_level; i++){

                auto levels = alg_hnsw->get_segTreelevel_2_LinksLevel(current_node_id,i);
                uint32_t* level = alg_hnsw->get_LinksLevel(levels,x);
                if(level==nullptr)
                    continue;
                int *data = (int*)alg_hnsw->get_linklist(current_node_id, *level);

                size_t size = alg_hnsw->getListCount((linklistsizeint*)data);
#ifdef USE_SSE
        _mm_prefetch(alg_hnsw->getDataByInternalId(*(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *) (data + 1), _MM_HINT_T0);
        _mm_prefetch((char *) (data + 1 ) + 64, _MM_HINT_T0);
#endif
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    uint32_t attr;
                    if( i != current_node_seglevel){
                        attr = attrs_hnsw[candidate_id];
                        if(attr < attr_l){
                            continue;
                        }else if(attr > attr_r){
                            break;
                        }
                    }
#ifdef USE_SSE
        _mm_prefetch(alg_hnsw->getDataByInternalId(*(data + j + 1)), _MM_HINT_T0);
#endif
                    if (!(visit_set->Test(candidate_id))) {
                        visit_set->Set(candidate_id);
                        dist_t dist = alg_hnsw->fstdistfunc_(data_point, alg_hnsw->getDataByInternalId(candidate_id), alg_hnsw->dist_func_param_);

                        //metric_dist_comps_++;
        
                        if (top_candidates.size() < k || lowerBound > dist ) {
                            if( i != current_node_seglevel){
                                for(auto &n : seglevel_and_range){
                                    if(n.left <= attr && n.right >= attr){
                                        PUSH_HEAP(candidate_set,dist, candidate_id, n.seglevel,n.x);
                                        PUSH_HEAP(top_candidates,dist, candidate_id);
                                        if (top_candidates.size() > k) {
                                            POP_HEAP(top_candidates);
                                        }
                                        
                                        lowerBound = TOP_HEAP(top_candidates).dist;
                                        break;
                                    }
                                }
                            }else{

                                PUSH_HEAP(candidate_set,dist, candidate_id, current_node_seglevel,current_node_x);
                                uint32_t attr = attrs_hnsw[candidate_id];
                                if(attr >= attr_l&&attr <= attr_r){
                                    PUSH_HEAP(top_candidates,dist, candidate_id);
                                    if (top_candidates.size() > k) {
                                        POP_HEAP(top_candidates);
                                    }
                                    lowerBound = TOP_HEAP(top_candidates).dist;
                                }
                            }
                        }
                    }
                }
                x=x/2;
            }
        }

        visit_set->Clear();

        while (top_candidates.size() > top_k) {
            POP_HEAP(top_candidates);
        }
        std::vector<u_int32_t> result;
        result.reserve(top_k);
        for (auto &i : top_candidates) {
            result.emplace_back(alg_hnsw->getExternalLabel(i.id));
        }
        return result;
    }


    template <typename T>
    void write(std::ofstream& ofs, const T& data) const {
        ofs.write(reinterpret_cast<const char*>(&data), sizeof(data));
    }

    template <typename T>
    void read(std::ifstream& ifs, T& data) {
        ifs.read(reinterpret_cast<char*>(&data), sizeof(data));
        if (!ifs) {
            throw std::runtime_error("file failed to read");
        }
    }

    void save(const std::string& filename) const {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("can not open file to write");
        }

        write(ofs, M);
        write(ofs, ef_construction);
        write(ofs, dim);
        write(ofs, max_elements_);
        write(ofs, treeNodeSize);
        write(ofs, cnt);
        write(ofs, l);
        write(ofs, r);
        write(ofs, root);
        write(ofs, segTreeMaxLevel);
        write(ofs, rebuildRatio);

        ofs.write(reinterpret_cast<const char*>(&treeNodeSize), sizeof(treeNodeSize));
        for (uint32_t i = 0; i < treeNodeSize; ++i) {
            const Node& node = tree[i];
            write(ofs, node.left);
            write(ofs, node.right);
            write(ofs, node.level);
            write(ofs, node.ep_id);
            write(ofs, node.size);
            write(ofs, node.true_elem_in_range_size);
            write(ofs, node.reuse);
        }
        ofs.write(reinterpret_cast<const char*>(attrs_hnsw), max_elements_ * sizeof(uint32_t));
        if (alg_hnsw) {
            alg_hnsw->saveIndex(filename + ".hnsw"); 
        }

    }



    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("can not open file to read");
        }

        read(ifs, M);
        read(ifs, ef_construction);
        read(ifs, dim); 
        read(ifs, max_elements_);
        read(ifs, treeNodeSize);
        read(ifs, cnt);
        read(ifs, l);
        read(ifs, r);
        read(ifs, root);
        read(ifs, segTreeMaxLevel);
        read(ifs, rebuildRatio);

        uint32_t loaded_tree_size;
        ifs.read(reinterpret_cast<char*>(&loaded_tree_size), sizeof(loaded_tree_size));
        tree = (Node *)malloc(loaded_tree_size * sizeof(Node));
        memset(tree, 0, loaded_tree_size * sizeof(Node));
        treeNodeSize = loaded_tree_size; 
        for (uint32_t i = 0; i < loaded_tree_size; ++i) {
            Node& node = tree[i];
            read(ifs, node.left);
            read(ifs, node.right);
            read(ifs, node.level);
            read(ifs, node.ep_id);
            read(ifs, node.size);
            read(ifs, node.true_elem_in_range_size);
            read(ifs, node.reuse);
        }
    
        std::string hnsw_filename = filename + ".hnsw";
        if (dim > 0) { 
            space = new hnswlib::L2Space(dim);  
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, hnsw_filename); 
        }
        visit_set =  new bitset_t<uint32_t>(max_elements_);

        attrs_hnsw = (uint32_t *)malloc(max_elements_ * sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(attrs_hnsw), max_elements_ * sizeof(uint32_t));
        alg_hnsw->attrs = attrs_hnsw;

        return ;
        


        struct NeighborWithAttr {
            int candidate_id;
            uint32_t attr;
            NeighborWithAttr(int a, uint32_t b):candidate_id(a),attr(b){}
            bool operator<(const NeighborWithAttr& other) const {
                return this->attr < other.attr;
            }
        };

        for(int i=0;i<alg_hnsw->cur_element_count;i++){

            unsigned int level = alg_hnsw->element_levels_[i];
            for (int lvl = 0; lvl < level; lvl++) { 
                int* data_lvl = (int*)alg_hnsw->get_linklist(i, lvl);
                size_t lvl_size = alg_hnsw->getListCount((linklistsizeint*)data_lvl);
                std::vector<NeighborWithAttr> neighbors_lvl;
                neighbors_lvl.reserve(lvl_size);

                for (size_t j = 1; j <= lvl_size; j++) {
                    int candidate_id = *(data_lvl + j);
                    uint32_t attr = attrs_hnsw[candidate_id];
                    neighbors_lvl.emplace_back(candidate_id, attr);
                }

                std::sort(neighbors_lvl.begin(), neighbors_lvl.end());

                *(linklistsizeint*)data_lvl = static_cast<linklistsizeint>(neighbors_lvl.size());
                for (size_t j = 0; j < neighbors_lvl.size(); j++) {
                    *(data_lvl + 1 + j) = neighbors_lvl[j].candidate_id;
                }
            }
        }
    }

};

