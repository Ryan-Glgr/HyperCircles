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
    HyperCircle(float rad, float *center, int cls);

    // finds the nearest neighbor to each HC
    void findNearestNeighbor(std::vector<Point> &dataSet);

    // creates our list of HC's
    static std::vector<HyperCircle> createCircles(std::vector<Point> &dataset);

    // function which takes all our built circles, and starts deleting them as possible.
    static void mergeCircles(std::vector<HyperCircle> &circles, std::vector<Point> &data);

    // wrapper function which makes all our circles by finding neighbors, then runs the merging algorithm and returns us our circles list
    static std::vector<HyperCircle> generateHyperCircles(std::vector<Point> &data);

    // helper function which checks if a given HC has a point inside it
    bool insideCircle(float *dataToCheck);

    static int classifyPoint(std::vector<HyperCircle> &circles, float *dataToCheck, int classificationMode, int numClasses);
    // used for the different classification types
    enum {
        SIMPLE_MAJORITY = 0,
        DENSITY_VOTING = 1,
        TOTAL_POINT_COUNT = 2
    };

};

#endif //HYPERCIRCLE_H
