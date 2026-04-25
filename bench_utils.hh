#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <random>
#include <numeric>
#include "./hnswlib/hnswlib.h"

namespace benchmark {
auto fvecs_read(const std::string &filename, size_t &d_out, size_t &n_out) -> float *
{
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + filename);
  }
  int d;
  in.read(reinterpret_cast<char *>(&d), 4);
  d_out = d;
  // calculate file size
  in.seekg(0, std::ios::beg);
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n    = file_size / (4 + d * sizeof(float));
  n_out       = n;
  float *data = new float[d * n];
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&d), 4);
    in.read(reinterpret_cast<char *>(data + i * d), d * sizeof(float));
  }
  in.close();
  return data;
}

auto LoadRange(const std::string &location) -> std::vector<std::pair<int,int>>
{
  std::vector<std::pair<int,int>> query_filters;
  std::ifstream                       ifs(location, std::ios::binary);
  if (!ifs.is_open()) {
    std::cout << "Fail to open: " << location << std::endl;
    std::abort();
  }
  // check meta size
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  auto n_query = file_size / (2 * sizeof(int));
  query_filters.resize(n_query);
  for (size_t i = 0; i < n_query; ++i) {
    int l, u;
    ifs.read(reinterpret_cast<char *>(&l), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&u), sizeof(int));
    query_filters[i] = std::pair<int,int>{l, u};
  }
  ifs.close();
  // LOG first 10 query filters
  std::string qf_str;
  for (size_t i = 0; i < std::min<size_t>(10, n_query); ++i) {
    qf_str += "[" + std::to_string(query_filters[i].first) + "," + std::to_string(query_filters[i].second) + "]";
  }
  std::cout << "First 10 query filters: " << qf_str << std::endl;
  //int x;cin>>x;
  return query_filters;
}


auto LoadGroundTruth(const std::string &gt_file) -> std::vector<std::vector<uint32_t>>
{
  std::ifstream in(gt_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + gt_file);
  }
  std::vector<std::vector<uint32_t>> all_gt;
  while (!in.eof()) {
    int k;
    in.read(reinterpret_cast<char *>(&k), sizeof(int));
    if (in.eof()) {
      break;
    }
    std::vector<uint32_t> gt(k);
    for (int i = 0; i < k; ++i) {
      unsigned int ib;
      in.read(reinterpret_cast<char *>(&ib), sizeof(unsigned int));
      gt[i] = ib;
    }
    all_gt.emplace_back(gt);
  }

  std::cout << "Loaded ground truth: " << all_gt.size() << std::endl;
  std::cout << "Example: first query has " << all_gt[0].size() << std::endl;
  for (auto ib : all_gt[0]) {
    std::cout << ib << ",";
  }
  std::cout << std::endl;
  return all_gt;
}

template <typename att_t>
auto LoadAttVec(const std::string att_file) -> std::vector<att_t>
{
  std::ifstream in(att_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + att_file);
  }
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n = file_size / sizeof(att_t);
  if (file_size % sizeof(att_t) != 0) {
    throw std::runtime_error("File size is not a multiple of att_t");
  }
  std::vector<att_t> att_vec(n);
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&att_vec[i]), sizeof(att_t));
  }
  in.close();
  std::cout << "Loaded att_vec: " << att_file << ", size: " << n << std::endl;
  return att_vec;
}

std::vector<std::pair<int, int>> load_increment_update(const std::string& file) {
    std::vector<std::pair<int, int>> updates;
    FILE* f = fopen(file.c_str(), "rb");
    if (!f) {
        std::cerr << "Cannot open file " << file << std::endl;
        return updates;
    }

    int id, attr;
    while (fread(&id, sizeof(int), 1, f) == 1 && fread(&attr, sizeof(int), 1, f) == 1) {
        updates.emplace_back(id, attr);
    }
    fclose(f);
    std::cout << "load_increment_update:" << file << " | rows:" << updates.size() << std::endl;
    return updates;
}


auto CalculateRecall(int k,std::vector<std::vector<uint32_t>> &gt, std::vector<std::vector<uint32_t>> &res)
    -> float
{
  if (k > gt.size()) {
    throw std::runtime_error("k is larger than gt size");
  }
  size_t n       = std::min(gt.size(), res.size());
  size_t total   = 0;
  size_t correct = 0;
  for (size_t i = 0; i < n; ++i) {
    if(gt[i].size()<k){
        total += gt[i].size();
    }else
        total += k;
    for (auto ib : res[i]) {
      if(gt[i].size()<k){
        //throw std::runtime_error("k is larger than gt size");
        if (std::find(gt[i].begin(), gt[i].begin()+gt[i].size(), ib) != gt[i].begin()+gt[i].size()) {
          correct++;
        }
      }
      else if (std::find(gt[i].begin(), gt[i].begin()+k, ib) != gt[i].begin()+k) {
        correct++;
      }
    }
  }
  return static_cast<float>(correct) / (total);
}

double get_current_memory_usage_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return (double)pmc.PrivateUsage / (1024 * 1024);
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return (double)usage.ru_maxrss / 1024;
    }
#endif
    return 0.0;
}


}  // namespace benchmark