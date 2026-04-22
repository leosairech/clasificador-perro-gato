#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: preprocess <input> <output>\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output = argv[2];

    cv::Mat img = cv::imread(input);
    if (img.empty()) {
        std::cerr << "Error al cargar imagen\n";
        return 1;
    }

    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::ofstream file(output);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            cv::Vec3f p = img.at<cv::Vec3f>(i, j);
            file << p[0] << "," << p[1] << "," << p[2];
            if (!(i == img.rows-1 && j == img.cols-1)) file << ",";
        }
    }

    file.close();
    return 0;
}
