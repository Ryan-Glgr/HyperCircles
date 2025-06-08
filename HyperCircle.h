//
// Created by Ryan Gallagher on 5/27/25.
//

#ifndef HYPERCIRCLE_H
#define HYPERCIRCLE_H

#include <vector>
#include <future>
#include <thread>
#include <algorithm>
#include <functional>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <map>
#include <queue>
#include <utility>


#include "Point.h"

class HyperCircle {

public:

    float radius;

    float *centerPoint;

    int classification;

    int numPoints;

    static std::vector<int> numCirclesPerClass;

    HyperCircle();
    HyperCircle(float rad, float *center, int cls);

    // finds the nearest neighbor to each HC
    void findNearestNeighbor(std::vector<Point> &dataSet);

    // similar to find nearest neighbor based creation. but this time it makes each circle as big as possible. and we are going to kill circle which are useless like normal.
    void findMaxDistance(std::vector<Point> &dataSet);

    // creates our list of HC's
    static std::vector<HyperCircle> createCircles(std::vector<Point> &dataset);

    // function which takes all our built circles, and starts deleting them as possible.
    static void mergeCircles(std::vector<HyperCircle> &circles, std::vector<Point> &dataSet);

    // wrapper function which makes all our circles by finding neighbors, then runs the merging algorithm and returns us our circles list
    static std::vector<HyperCircle> generateHyperCircles(std::vector<Point> &dataSet, int numClasses);

    static std::vector<HyperCircle> generateMaxDistanceBasedHyperCircles(std::vector<Point> &dataSet, int numClasses);

    static void removeUselessCircles(std::vector<HyperCircle> &circles, std::vector<Point> &dataSet);

    // helper function which checks if a given HC has a point inside it
    bool insideCircle(float *dataToCheck);

    // classification mode determines whether we use HC's or KNN (or whatever other fallback). then we use the sub mode in the switch to determine voting style or which particular fallback
    static int classifyPoint(std::vector<HyperCircle> &circles, std::vector<Point> &train, float *dataToCheck, int classificationMode, int subMode,  int numClasses, int k);

    // all the different ways we can use the HC's for voting on each class
    enum {
        SIMPLE_MAJORITY = 0,
        COUNT_VOTE = 1,
        DENSITY_VOTE = 2,
        DISTANCE_VOTE = 3,
        PER_CLASS_VOTE = 4,
        SMALLEST_CIRCLE = 5
    };

    // used for the different fallback types
    enum {
        USE_CIRCLES = 0, // normal version, where we are not using fallback.
        REGULAR_KNN = 1,
        K_NEAREST_CIRCLES = 2,
        K_NEAREST_RATIOS = 3,
    };

    static int regularKNN(std::vector<Point> &dataSet, float *point, int k, int numClasses);

    static int kNearestCircle(std::vector<HyperCircle> &circles, float *point, int k, int numClasses);

    static int kNearestCircleRatio(std::vector<HyperCircle> &circles, float *point, int k, int numClasses);

};

#endif //HYPERCIRCLE_H
