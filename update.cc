#include "segtree.h"
#include "bench_utils.hh"

#include <map>
#include <unordered_map>
#include <iomanip>


int main(int argc, char **argv)
{
    std::string index_location;
    std::string update_file; 
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--index_location") == 0) {
        index_location = argv[++i];
        } else if (strcmp(argv[i], "--update_file") == 0) {
        update_file = argv[++i];
        }else{
        throw std::runtime_error("unknown argument: " + std::string(argv[i]));
        }
    }

    std::cout << "update_file: " << update_file
                << ", index_location: " << index_location << std::endl;


    SegmentTree tree;
    tree.load(index_location);

    auto updates = benchmark::load_increment_update(update_file);
            
    auto update_start = std::chrono::high_resolution_clock::now();
    for (auto& [vid, new_attr] : updates) {
        tree.updateOneLabel(new_attr,vid);
    }
    auto update_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> update_elapsed = update_end - update_start;

    std::cout << "update time: " << update_elapsed.count() << " s" << std::endl;
    tree.save(index_location);
    return 0;
}
      