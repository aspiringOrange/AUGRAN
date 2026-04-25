#include "segtree.h"
#include "bench_utils.hh"

#include <map>
#include <unordered_map>
#include <iomanip>


const size_t START_EF = 10;           
const size_t INIT_STEP = 200;          
const float TARGET_RECALL_MIN = 0.80f;   
const float TARGET_RECALL_MAX = 0.99f;  
const float TARGET_RECALL_STEP = 0.01f;    
const float RECALL_UPPER_BOUND = 0.99f;   
const size_t MAX_EF_LIMIT = 10011; 

struct EfMetrics {
    float recall;               
    float qps;                 
    size_t avg_dist_comps;      
    size_t avg_hops;        
    float total_time;         
};

float query_ef(size_t efs, size_t k, size_t d, size_t nq, float* query_vecs,
               SegmentTree& index,
               const std::vector<std::pair<int,int>>& query_filters,
                std::vector<std::vector<uint32_t>>& gt,
               std::map<size_t, EfMetrics>& ef_metrics_map) {

    std::vector<std::vector<uint32_t>> results(nq);
    float total_time = 0.0f;
    index.metric_dist_comps_ = 0;


    for (size_t i = 0; i < nq; ++i) {

        auto start = std::chrono::high_resolution_clock::now();
        auto result = index.get_nearest_neighbors(query_vecs + i * d,query_filters[i].first,query_filters[i].second,efs,k);
        auto end   = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration<float>(end - start).count();
        for(auto &r : result) {
            results[i].emplace_back(r);
        }
    }

    EfMetrics metrics;
    metrics.recall = benchmark::CalculateRecall(k, gt, results);
    metrics.qps = (total_time > 0) ? static_cast<float>(nq) / total_time : 0.0f;
    metrics.avg_dist_comps = index.metric_dist_comps_ / nq;
    metrics.avg_hops = 0/ nq;
    metrics.total_time = total_time;

    if (ef_metrics_map.find(efs) == ef_metrics_map.end()) {
        ef_metrics_map[efs] = metrics;
        // std::cout << std::fixed << std::setprecision(4);
        // std::cout << "[Explored EF=" << efs << "] "
        //           << "Recall=" << metrics.recall << " | "
        //           << "QPS=" << metrics.qps << " | "
        //           << "AvgDistComps=" << metrics.avg_dist_comps << " | "
        //           << "AvgHops=" << metrics.avg_hops << std::endl;
    }

    return metrics.recall;
}

size_t auto_explore_ef(size_t k, size_t d, size_t nq, float* query_vecs,
                       SegmentTree& index,
                       const std::vector<std::pair<int,int>>& query_filters,
                std::vector<std::vector<uint32_t>>& gt,
               std::map<size_t, EfMetrics>& ef_metrics_map) {
    std::cout << "\n===== Start Auto Explore EF from " << START_EF << " =====" << std::endl;
    size_t current_ef = START_EF;
    float current_recall = 0.0f;

    while (current_recall < RECALL_UPPER_BOUND && current_ef < MAX_EF_LIMIT) {
        current_recall = query_ef(current_ef, k, d, nq, query_vecs, index, query_filters, gt, ef_metrics_map);
        current_ef += INIT_STEP;
    }

    size_t max_ef = current_ef - INIT_STEP;
    std::cout << "===== Auto Explore Finished, EF Range [10, " << max_ef 
              << "], Max Recall: " << current_recall << " =====" << std::endl;
    return max_ef;
}

size_t binary_search_best_ef(float target_recall, size_t min_ef, size_t max_ef,
                             size_t k, size_t d, size_t nq, float* query_vecs,
                             SegmentTree& index,
                       const std::vector<std::pair<int,int>>& query_filters,
                std::vector<std::vector<uint32_t>>& gt,
               std::map<size_t, EfMetrics>& ef_metrics_map) {
    size_t best_ef = min_ef;
    float min_diff = fabs(ef_metrics_map[min_ef].recall - target_recall);

    size_t left = min_ef;
    size_t right = max_ef;
    if (ef_metrics_map.find(left) != ef_metrics_map.end()) {
            if(ef_metrics_map[left].recall > target_recall)
                return left;
    }
    if (ef_metrics_map.find(right) != ef_metrics_map.end()) {
            if(ef_metrics_map[right].recall < target_recall)
                return right;
    }
    while (left <= right) {
        size_t mid = left + (right - left) / 2; 
        float mid_recall;

        if (ef_metrics_map.find(mid) != ef_metrics_map.end()) {
            mid_recall = ef_metrics_map[mid].recall;
        } else {
            mid_recall = query_ef(mid, k, d, nq, query_vecs, index, query_filters, gt, ef_metrics_map);
        }

        float current_diff = fabs(mid_recall - target_recall);
        if (current_diff < min_diff) {
            min_diff = current_diff;
            best_ef = mid;
        } else if (current_diff == min_diff) {
            best_ef = std::min(best_ef, mid);
        }

        if (mid_recall < target_recall) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return best_ef;
}
int main(int argc, char **argv)
{
  std::string quer_vec, query_rng, gt_file, index_location;
  size_t      k;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--query_vec") == 0) {
      quer_vec = argv[++i];
    } else if (strcmp(argv[i], "--query_rng") == 0) {
      query_rng = argv[++i];
    } else if (strcmp(argv[i], "--gt_file") == 0) {
      gt_file = argv[++i];
    } else if (strcmp(argv[i], "--k") == 0) {
      k = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--index_location") == 0) {
      index_location = argv[++i];
    }else{
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }

  std::cout << "query_vec: " << quer_vec << ", query_rng: " << query_rng << ", gt_file: " << gt_file << ", k: " << k
            << ", index_location: " << index_location << std::endl;


  SegmentTree tree;
  tree.load(index_location);
  cout<<"Index space:"<<benchmark::get_current_memory_usage_mb()<<"MB"<<endl;
  size_t d, nq;
  float *query_vecs = benchmark::fvecs_read(quer_vec, d, nq);
  std::cout << "Loaded query vectors: " << quer_vec << ", d: " << d << ", nq: " << nq << std::endl;
  // load query filters
  std::vector<std::pair<int,int>> query_filters = benchmark::LoadRange(query_rng);
  std::cout << "Loaded query filters: " << query_rng << std::endl;

  // load ground truth
  std::vector<std::vector<uint32_t>> gt = benchmark::LoadGroundTruth(gt_file);
  std::cout << "Loaded ground truth: " << gt_file << std::endl;
  nq = gt.size();

  std::cout << "searching..." << std::endl;


  // std::map<size_t, EfMetrics> ef_metrics_map; 
  // std::map<float, std::pair<size_t, EfMetrics>> target_result_map; 

  // size_t max_ef = auto_explore_ef(k, d, nq, query_vecs, tree, query_filters, gt, ef_metrics_map);

  // std::cout << "\n===== Target Recall (0.80~0.99) -> Nearest EF =====" << std::endl;
  // std::cout << std::fixed << std::setprecision(4);
  // std::cout << "TargetRecall,NearestEF,Recall,QPS" << std::endl;

  // for (float target = TARGET_RECALL_MIN; target <= TARGET_RECALL_MAX; target += TARGET_RECALL_STEP) {
  //   target = round(target * 100) / 100;
  //   size_t best_ef = binary_search_best_ef(target, START_EF, max_ef,
  //                                              k, d, nq, query_vecs, tree, query_filters, gt, ef_metrics_map);
  //   const EfMetrics& best_metrics = ef_metrics_map[best_ef];
  //   target_result_map[target] = {best_ef, best_metrics};
  //   std::cout << target << "," << best_ef 
  //                  << "," << best_metrics.recall << ","
  //                 << best_metrics.qps<< std::endl;
  // }

  // std::cout << "search done" << std::endl;
  // return 0;


  std::vector<size_t> efs_list{23,11,12,13,14,15,20,30,40,50,60,80,100,200,300};

  for(auto efs: efs_list){
    float max_qps=0;
    float r=0;
    for(int m=0;m<1;m++){
    std::vector<std::vector<uint32_t>> results(nq);
    float time = 0;
    float avg_dist = 0;
    float avg_hops = 0;
    tree.metric_dist_comps_ = 0;
    vector<int> idxs;
    
    for (size_t i = 0; i < nq; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      auto result = tree.get_nearest_neighbors(query_vecs + i * d,query_filters[i].first,query_filters[i].second,efs,k);
      auto end   = std::chrono::high_resolution_clock::now();
      time += std::chrono::duration<float>(end - start).count();
      for(auto &r : result) {
        results[i].emplace_back(r);
      }   
    }
    float recall = benchmark::CalculateRecall(k,gt,results);
    r= recall;
    max_qps = max(max_qps,nq/time);
  }
    cout<<r<<" "<<max_qps<<endl;
  }
  std::cout << "search done" << std::endl;
  return 0;
}