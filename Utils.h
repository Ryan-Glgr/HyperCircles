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


class Utils {
public:

    // helper function which just gets us our distance so we don't have to write it all over.
    static float euclideanDistance(float *a, float *b, int FIELD_LENGTH) {
        float sum = 0.0f;
        for (int i = 0; i < FIELD_LENGTH; i++) {
            float diff = a[i] - b[i];
            diff *= diff;
            sum += diff;
        }
        return sqrt(sum);
    }

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
        for (auto& [cls, points] : classBuckets) {
            shuffle(points.begin(), points.end(), rng);
            for (int i = 0; i < points.size(); ++i) {
                folds[i % k].push_back(points[i]);
            }
        }

        return folds;
    }

};



#endif //UTILS_H
