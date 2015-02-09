#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <boost/program_options.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/legacy/legacy.hpp>

using namespace cv;
using namespace std;

#define DEFAULT_MIN_WIDTH 300
#define DEFAULT_MIN_HEIGHT 300
#define DEFAULT_MIN_DESC 100
#define HESSIAN_THRESHOLD 300

enum return_error_code {
    NO_ERROR,
    MISSING_INPUT_FILE,
    UNABLE_TO_READ_FILE,
    IMAGE_TOO_SMALL,
    NOT_ENOUGH_DESCRIPTORS
};

int main(int argc, char* argv[]) {

    // set some default parameters
    int min_width = DEFAULT_MIN_WIDTH;
    int min_height = DEFAULT_MIN_HEIGHT;
    int min_desc = DEFAULT_MIN_DESC;
    std::string input_file;

    // Define and parse the program option
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
            ("help,h", "print help messages")
            ("display,d", "display keypoints")
            ("width,W", po::value<int>(&min_width), "minimum width")
            ("height,H", po::value<int>(&min_height), "minimum height")
            ("descriptors,D", po::value<int>(&min_desc), "minimum number of descriptors")
            ("output-file,o", po::value<std::string>(), "save input image keypoints")
            ("input-file", po::value<std::string>(), "input file");

    po::positional_options_description p;
    p.add("input-file", -1);

    po::variables_map vm;

    try {
        po::store(po::command_line_parser(argc, argv).
                options(desc).positional(p).run(), vm);
        vm.notify();
    }
    catch (boost::program_options::error& e) {
        std::cerr << "Input Exception: " << e.what() << std::endl;
    }

    std::cerr << "width: " << min_width << ", height: " << min_height << ", min_desc: " << min_desc << std::endl;

    if(vm.count("help")) {
        std::cout << "USAGE: ./surf_detector [OPTIONS] input_file" << std::endl;
        std::cout << desc << std::endl;
        exit(0);
    }

    if(vm.count("input-file")) {
        input_file = vm["input-file"].as<std::string>();
    }
    else {
        std::cout << MISSING_INPUT_FILE << std::endl;
        exit(-1);
    }

    cv::Mat input_image = cv::imread(input_file, CV_LOAD_IMAGE_GRAYSCALE);

    //check for the data read error
    if(!input_image.data) {
        std::cout << UNABLE_TO_READ_FILE << std::endl;
        exit(-1);
    }

    if(input_image.cols < min_width || input_image.rows < min_height) {
        std::cout << IMAGE_TOO_SMALL << std::endl;
        exit(-1);
    }

    //detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    cv::SurfFeatureDetector surf(HESSIAN_THRESHOLD);
    surf.detect(input_image, keypoints);

    //check the number of descriptors
    if(static_cast<int>(keypoints.size()) < min_desc) {
        std::cout << NOT_ENOUGH_DESCRIPTORS << std::endl;
        exit(-1);
    }

    //draw the keypoints
    if(vm.count("display")) {
        cv::drawKeypoints(input_image, keypoints, input_image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("features", input_image);
        cv::waitKey(0);
    }

    //write the results
    if(vm.count("output-file")) {
        std::string output_file = vm["output-file"].as<std::string>();
        cv::imwrite(output_file, input_image);
    }

    std::cout << NO_ERROR << std::endl;
}

