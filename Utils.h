//
// Created by Ryan Gallagher on 5/27/25.
//

#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <iostream>
#include <limits>
#include <cstdlib>
#include "Point.h"
#include <random>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <cstddef>

class Utils {
public:

#define NORM 2

#if NORM == 1
    // a simple manhattan distance, which may be better for pictures, but is not a true "circle". it's a diamond or rhombus in shape
    static inline float distance(const float *__restrict a, const float * __restrict b, const int n) {
        float sum = 0.0f;

        // get the largest number we can iterate through to in our unrolled loops
        const int limit = n & ~3;

#pragma omp simd reduction(+:sum)
        for (int i = 0; i < limit; i += 4) {
            sum += std::fabs(a[i] - b[i]);
            sum += std::fabs(a[i+1] - b[i+1]);
            sum += std::fabs(a[i+2] - b[i+2]);
            sum += std::fabs(a[i+3] - b[i+3]);
        }

        // handle remaining stuff
        for (int i = limit; i < n; ++i) {
            sum += std::fabs(a[i] - b[i]);
        }
        return sum;
    }
#elif NORM == 2
    // just a simple euclidean distance measure
    // we use restrict *'s and we use simd to vectorize this operation and do it FAST
    static inline float distance(const float* __restrict a, const float* __restrict b, const int n) {
        float sum = 0.0f;
        const int limit = n & ~3;  // for 4-wide unroll. gets us the largest multiple of 4 <= N.

        // we are going to do a reduction, and we have manually unrolled the loop here so that we do less operations.
        // vectorized operations
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < limit; i += 4) {
            float d0 = a[i] - b[i];
            float d1 = a[i+1] - b[i+1];
            float d2 = a[i+2] - b[i+2];
            float d3 = a[i+3] - b[i+3];
            sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
        // get the remaining values.
        for (int i = limit; i < n; ++i) {
            float d = a[i] - b[i];
            sum += d*d;
        }

        return sqrt(sum);
    }
#else
    // L3 distance: cube root of the sum of cubed absolute differences
    static inline float distance(const float* __restrict a, const float* __restrict b, const int n) {
        float sum = 0.0f;
        const int limit = n & ~3;

        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < limit; i += 4) {
            float d0 = fabsf(a[i] - b[i]);
            float d1 = fabsf(a[i+1] - b[i+1]);
            float d2 = fabsf(a[i+2] - b[i+2]);
            float d3 = fabsf(a[i+3] - b[i+3]);
            sum += d0*d0*d0 + d1*d1*d1 + d2*d2*d2 + d3*d3*d3;
        }

        for (int i = limit; i < n; ++i) {
            float d = fabsf(a[i] - b[i]);
            sum += d*d*d;
        }
        return cbrtf(sum);
    }
#endif

    static void waitForEnter() {
        std::cout << "\nPress Enter to continue...";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // Function to display the main menu
    static void displayMainMenu() {
#ifdef _WIN32
        system("cls");
#else
        system("clear");
#endif
        std::cout << "=== HyperCircle Classification System ===\n\n";
        std::cout << "1. Import training data.\n";
        std::cout << "2. Import testing data.\n";
        std::cout << std::endl;
        std::cout << "3. Generate HyperCircles using nearest neighbor and merging.\n";
        std::cout << "4. Generate HyperCircles using max radius search.\n";
        std::cout << std::endl;
        std::cout << "5. Test HyperCircles.\n";
        std::cout << "6. K Fold Cross Validation.\n";
        std::cout << std::endl;
        std::cout << "7. Save HC's to a file\n";
        std::cout << "8. Load HC's from a file\n";
        std::cout << std::endl;
        std::cout << "9. Find Best HC voting on test data.\n";
        std::cout << "10. Find Best KNN mode on test data.\n";
        std::cout << std::endl << std::endl;
        std::cout << "-1. Exit\n";
    }

    // Stratified k-fold split
    static std::vector<std::vector<Point>> stratifiedKFolds(int k, std::vector<Point> &data, int seed = 42) {
        // Map from class ID to all points of that class
        std::unordered_map<int, std::vector<Point>> classBuckets;
        for (const auto& point : data) {
            classBuckets[point.classification].push_back(point);
        }

        // Set up RNG
        std::mt19937 rng(seed);

        // Create k empty folds
        std::vector<std::vector<Point>> folds(k);

        // Distribute each class's points across folds
        for (auto& entry : classBuckets) {
            auto& points = entry.second;

            shuffle(points.begin(), points.end(), rng);
            for (int i = 0; i < points.size(); ++i) {
                folds[i % k].push_back(points[i]);
            }
        }

        return folds;
    }

};

#endif //UTILS_H
