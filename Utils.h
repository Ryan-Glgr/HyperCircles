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

#define SLOW_DISTANCE 0
#if SLOW_DISTANCE
    // helper function which just gets us our distance so we don't have to write it all over.
    static inline float euclideanDistance(float *a, float *b, int FIELD_LENGTH) {
        float sum = 0.0f;
        for (int i = 0; i < FIELD_LENGTH; i++) {
            float diff = a[i] - b[i];
            diff *= diff;
            sum += diff;
        }
        return sqrt(sum);
    }
#else

    // same as above, but super optimized.
    // we use restrict *'s and we use simd to vectorize this operation and do it FAST
    static inline float euclideanDistance(const float* __restrict a, const float* __restrict b, const int n) {
        float sum = 0.0f;
        const int limit = n & ~3;  // for 4-wide unroll. gets us the largest multiple of 4 <= N.

        // we are going to do a reduction, and we have manually unrolled the loop here so that we do less operations.
        // vectorized operations
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i <= limit; i += 4) {
            float d0 = a[i] - b[i];
            float d1 = a[i+1] - b[i+1];
            float d2 = a[i+2] - b[i+2];
            float d3 = a[i+3] - b[i+3];
            sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
        // get the remaining values.
        for (int i = limit + 4; i < n; ++i) {
            float d = a[i] - b[i];
            sum += d*d;
        }

        return sqrt(sum);
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
        std::cout << "3. Generate HyperCircles.\n";
        std::cout << "4. Test HyperCircles.\n";
        std::cout << "5. K Fold Cross Validation.\n\n";
        std::cout << std::endl;
        std::cout << "6. Quit\n\n";
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
