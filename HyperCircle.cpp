#include "HyperCircle.h"
#include "Utils.h"
using namespace std;

HyperCircle::HyperCircle() {
    radius = 0.0f;
    centerPoint = nullptr;
    classification = -1;
    numPoints = -1;
}

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
    vector<HyperCircle> circles(dataset.size());

    // Parallel HC creation.
    #pragma omp parallel for
    for (int i = 0; i < dataset.size(); ++i) {
        Point &p = dataset[i];
        circles[i] = HyperCircle(0.0f, p.location, p.classification);
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
// function which takes all our built circles, and starts deleting them as possible.
void HyperCircle::mergeCircles(vector<HyperCircle>& circles, vector<Point>& data) {

    for (int idx = 0; idx < circles.size(); ++idx) {

        // skip circles we've already eaten
        if (circles[idx].centerPoint == nullptr)
            continue;

        auto& c = circles[idx];

        // get all our distances. pair is distance, circleID
        vector<pair<float,int>> dists;
        // pre allocate the vector as the right size so we can save the copies
        dists.reserve(circles.size());

        #pragma omp parallel
        {
            // our own local copy of the circles pairs
            vector<pair<float,int>> local;
            local.reserve(data.size() / omp_get_num_threads());

            // use nowait so that they don't bother synchronizing
            #pragma omp for nowait
            for (int j = idx + 1; j < circles.size(); ++j) {

                // skip dead or wrong-class circles
                if (circles[j].centerPoint == nullptr || circles[j].classification != c.classification)
                    continue;

                // get our distance between our two circles. push that plus smaller guy radius.
                local.emplace_back(Utils::euclideanDistance(c.centerPoint,circles[j].centerPoint, Point::numAttributes) + circles[j].radius, j);
            }

            // add all our distances
            #pragma omp critical
            dists.insert(dists.end(), local.begin(), local.end());
        }

        // sort the circles closest to biggest.
        // remember that to be mergeable, we have to be able to EAT the distance between our radius and theirs.
        sort(dists.begin(), dists.end());   // cheapest pair first

        vector<int> mergable;
        mergable.reserve(dists.size());

        // now iterate through our sorted list, eating all the circles we can.
        for (const auto& smallestDistance : dists) {

            // get our info about this circle
            float newR2 = smallestDistance.first;
            int   circleID = smallestDistance.second;

            // if the distance is inside our existing radius, we can skip and mark this guy as eaten.
            if (newR2 <= c.radius) {
                mergable.push_back(circleID);
                continue;
            }

            volatile int canMerge = 1; // shared flag
            #pragma omp parallel for schedule(static) shared(canMerge)
            for (int p = 0; p < data.size(); ++p) {

                if (!canMerge)
                    continue;   // early-out after failure

                const Point& pt = data[p];
                if (pt.classification == c.classification)
                    continue;

                if (Utils::euclideanDistance(pt.location,c.centerPoint, Point::numAttributes) <= newR2) {

                    #pragma omp atomic write
                    canMerge = 0;

                    #pragma omp cancel for
                }
                #pragma omp cancellation point for
            }

            if (canMerge) {
                c.radius = newR2;
                mergable.push_back(circleID);   // remember the real ID
            } else
                break;
        }

        // mark every circle we “ate” as null instead of erasing now
        for (int id : mergable)
            circles[id].centerPoint = nullptr;

    }

    // single compaction pass – remove all null centerpoints HC's
    circles.erase(remove_if(circles.begin(), circles.end(),[](const HyperCircle& hc){return hc.centerPoint == nullptr; }), circles.end());
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

            int *votes = new int[numClasses]();

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

