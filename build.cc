#include "segtree.h"
#include "bench_utils.hh"


int main(int argc, char **argv)
{

  size_t      m, efc, dim, maxN;
  std::string basevec, baseatt, index_location;
  float rebuildRatio;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--m") == 0) {
      m = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--efc") == 0) {
      efc = std::stoul(argv[++i]);
    } else if (strcmp(argv[i], "--basevec") == 0) {
      basevec = argv[++i];
    } else if (strcmp(argv[i], "--baseatt") == 0) {
      baseatt = argv[++i];
    } else if (strcmp(argv[i], "--index_location") == 0) {
      index_location = argv[++i];
    } else if (strcmp(argv[i], "--rebuild_ratio") == 0) {
      rebuildRatio = std::stof(argv[++i]);
    } else {
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
    }
  }
  std::cout << "m: " << m << ", efc: " << efc << ", basevec: " << basevec << ", baseatt: " << baseatt << ", rebuild_ratio: " << rebuildRatio << std::endl;
  // load base vectors
  float *basevecs = benchmark::fvecs_read(basevec, dim, maxN);
  SegmentTree tree;
  tree.init(dim,maxN,m,efc,rebuildRatio);

  std::vector<int> ids(maxN);
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), std::mt19937{std::random_device{}()});
  std::vector<int> att_vec = benchmark::LoadAttVec<int>(baseatt);
  
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < maxN; ++i) {
    cout<<"\r"<<i<<"/"<<maxN;
    auto cur_id = ids[i];
    tree.insertOne(att_vec[cur_id], basevecs + cur_id * dim,cur_id);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Index built in " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
  // save index
  tree.save(index_location);
  std::cout << "Index saved to: " << index_location << std::endl;

}