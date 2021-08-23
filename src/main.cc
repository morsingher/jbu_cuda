#include "upsampling.h"

int main(int argc, char** argv) {

    if (argc < 4) {
        std::cout << "Usage: <executable> path_in_LR path_in_HR path_out" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string filename_lr(argv[1]);
    const std::string filename_hr(argv[2]);
    const std::string out_folder(argv[3]);

    if (!JointBilateralUpsampling(filename_lr, filename_hr, out_folder)) {
        std::cout << "Failed to upsample!" << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
