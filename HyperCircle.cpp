//
// Created by Ryan Gallagher on 5/27/25.
//

#include "HyperCircle.h"
#include "Utils.h"
using namespace std;

HyperCircle::HyperCircle(float rad, float *center, int cls) {
    radius = rad;
    centerPoint = center;
    classification = cls;
    numPoints = 1;
}

// finds the nearest neighbor to each HC
// this is useful, so that we can get the distance to each nearest neighbor and update the radius.
void HyperCircle::findNearestNeighbor(vector<Point> &dataSet) {

    // find our nearest guy of our own class, and set our radius to that value
    float minDist = numeric_limits<float>::max();
    int minClass = -1;
    for (auto &p : dataSet) {

        // if this point is the one which made our HC, continue.
        if (p.location == this->centerPoint) {
            continue;
        }

        float newDist = Utils::euclideanDistance(p.location, centerPoint, Point::numAttributes);

        // using <= so that if we had a TIE, we would update classification. this might help us catch errors
        if (newDist < minDist) {
            minDist = newDist;
            minClass = p.classification;
        }
        else if (newDist == minDist) {
            // if the class right now is our own, we set it to whatever the point has. this allows us
            // to have a tie goes to wrong class behavior. this way if two points are same distance, we
            // don't let in guy from wrong class.
            if (minClass == this->classification) {
                minClass = p.classification;
            }
        }
    }

    // if our nearest point was this class, we use the distance to it as our radius. if not, we need to leave radius as 0.0
    if (minClass == this->classification) {
        radius = minDist;
    }
}

// takes in the entire dataset, split up by class
vector<HyperCircle> HyperCircle::createCircles(vector<Point> &dataset) {

    // make a circle out of each point
    vector<HyperCircle> circles;
    for (auto &p : dataset) {
        circles.push_back(HyperCircle(0.0, p.location, p.classification));
    }

    // update each circle's radius to our nearest neighbor.
    #pragma omp parallel for
    for (int c = 0; c < circles.size(); c++) {
        auto &circle = circles[c];
        circle.findNearestNeighbor(dataset);
    }

    return circles;
}

// function which takes all our built circles, and starts deleting them as possible.
void HyperCircle::mergeCircles(vector<HyperCircle> &circles, vector<Point> &data) {

    // this function is complex. there are several ways to perform the merging process. The simplest
    // would be to just consider if we can absorb another neighbors closest neighbor. that is, add their radius to the distance between
    // our two centerpoints without allowing any wrong class points into the circle
    // a more complex version to merge would be taking the maximum radius each point could expand to into account.
    // This may work a bit better, but can also be susceptible to outlier points.

    // simple version, where i just try and take my nearest neighbors nearest neighbor.
    // the easiest way to do this, just add the radius of small point to the current one, and then check if any wrong class points are in this class now.
    for (int circleIndex = 0; circleIndex < circles.size();) {

        // grab our outer circle
        auto &circle = circles[circleIndex];

        // this is going to store the distances to all the other circles.
        priority_queue<pair<float, int>, vector<pair<float, int>>, greater<std::pair<float, int>>> circleQueue;

        // insert the distance to each circle, and it's index in the circle queue.
        for (int smallerCircleIndex = circleIndex + 1; smallerCircleIndex < circles.size(); smallerCircleIndex++) {

            // don't bother pushing circles for the wrong classes.
            if (circles[smallerCircleIndex].classification != circle.classification)
                continue;

            // compute the distance of the two centers, plus the radius of other circle. if that is within our own radius, we know that we already have this point somehow.
            circleQueue.emplace(Utils::euclideanDistance(circle.centerPoint,circles[smallerCircleIndex].centerPoint, Point::numAttributes) + circles[smallerCircleIndex].radius, smallerCircleIndex);
        }

        set<int> circlesToDelete;
        while (!circleQueue.empty()) {

            pair<float, int> small = circleQueue.top();
            circleQueue.pop();

            // if the one we are trying to eat doesn't belong to our class. break.
            if (circles[small.second].classification != circle.classification) {
                break;
            }

            // if the radius is inside of our current radius, we know we can safely delete this circle.
            float newRadius = small.first;
            if (newRadius <= circle.radius) {
                circlesToDelete.insert(small.second);
                continue;
            }

            volatile int canMerge = 1;
            // do this in parallel, obviously. then we do a reduction using && operator so that we can combine all results.
            #pragma omp parallel for reduction(&& : canMerge)
            for (int p = 0; p < static_cast<int>(data.size()); ++p) {
                const auto &pt = data[p];

                if (pt.classification == circle.classification)
                    continue;

                if (Utils::euclideanDistance(pt.location, circle.centerPoint, Point::numAttributes) <= newRadius){
                    canMerge = 0;
                }
            }

            // if we can merge, we take the new radius we computed, and then we delete our smaller circle, since we have engulfed him.
            if (canMerge) {
                circle.radius = newRadius;
                circlesToDelete.insert(small.second);
            }
            else break;
        }
        // now we remove all those circles which we just flagged
        for (auto it = circlesToDelete.rbegin(); it != circlesToDelete.rend(); ++it) {
            circles.erase(circles.begin() + *it);
            if (*it < circleIndex) --circleIndex; // Adjust index if deletion was before current one
        }

        ++circleIndex;
    }
}

bool HyperCircle::insideCircle(float *dataToCheck) {
    return (Utils::euclideanDistance(centerPoint, dataToCheck, Point::numAttributes) <= radius);
}

// function which makes us a list of circles given some pre processed data
vector<HyperCircle> HyperCircle::generateHyperCircles(vector<Point> &data) {

    // generate our initial list of circles
    vector<HyperCircle> circles = createCircles(data);

    // merge our circles so that we can get larger circles
    mergeCircles(circles, data);


    for (auto &circle : circles) {

        int insideCount = 0;

        // do a reduction to efficiently compute the size of the circle in terms of point count
        // we do the reduction because we are computing a ton of euclidean distances in the inside function.
        #pragma omp parallel for reduction(+ : insideCount)
        for (int p = 0; p < data.size(); ++p) {

            auto &point = data[p];
            // if this point is inside, increment numPoints
            if (circle.insideCircle(point.location)) {
                insideCount++;
            }
        }
        // update the value now that the for loop is over
        circle.numPoints = insideCount;
    }
    return circles;
}

int HyperCircle::classifyPoint(vector<HyperCircle> &circles, float *dataToCheck, int classificationMode, int numClasses) {

    // here we use our different classification options.
    // first option is to just take whichever class we find our point in the most.
    // we could also use the average density of each circle, so number of points / area on average of each circle
    // another option is to use the total point count of each circle our point fell into
    int prediction = -1;
    switch (classificationMode) {

        case SIMPLE_MAJORITY: {

            int *votes = new int[numClasses];

            // for each HC, if the point is inside, we simply up the count.
            for (int i = 0; i < circles.size(); i++) {
                if (circles[i].insideCircle(dataToCheck)) {
                    votes[circles[i].classification]++;
                }
            }

            // start maxVotes at 0. that way if no votes are cast, we know that we have to classify with the fallback mechanisms
            int maxVotes = 0;
            for (int cls = 0; cls < numClasses; ++cls) {
                if (votes[cls] > maxVotes) {
                    maxVotes = votes[cls];
                    prediction = cls;
                }
            }

            delete[] votes;
        }


    }



    // fallback mechanism
    return prediction;
}
