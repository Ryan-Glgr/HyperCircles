#include "HyperCircle.h"
#include "Utils.h"
using namespace std;

// parameters we can play with.
#define MIN_RADIUS 0.0f

// tracks how many circles we have in each class.
vector<int> HyperCircle::numCirclesPerClass;

HyperCircle::HyperCircle() {
    radius = 0.0f;
    centerPoint = nullptr;
    maxPureDistance = 0.0f;
    classification = -1;
    numPoints = -1;
}

HyperCircle::HyperCircle(float rad, float *center, int cls) {
    radius = rad;
    centerPoint = center;
    maxPureDistance = rad;
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

        // take the distance
        float newDist = Utils::distance(p.location, centerPoint, Point::numAttributes);

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

    // if our nearest point was this class, we use the distance to it as our radius. if not, we need to leave radius as 0.0. if 0.0 we kill the circle later.
    if (minClass == this->classification) {
        radius = minDist;
    }
}

// similar to findNearestNeighbor. but this version finds the largest pure distance. this way we know exactly how big each circle can be.
// then we can set all the radiuses to said distance, and just remove useless circles. no merging needed.
void HyperCircle::findMaxDistance(vector<Point> &dataSet) {

    // store distance to all training points
    vector<pair<float, int>> distances(dataSet.size());

    // compute all those distances
    for (int dp = 0; dp < dataSet.size(); ++dp) {
        distances[dp] = {Utils::distance(dataSet[dp].location, this->centerPoint, Point::numAttributes), dataSet[dp].classification};
    }

    // sort all those distances.
    sort(distances.begin(), distances.end());

    // set our radius to the largest distance we can use while keeping pure classification
    float maxDistance = 0.0f;
    for (auto & distance : distances) {
        if (distance.second == this->classification) {
            maxPureDistance = maxDistance;
        }
        else break;
    }

}

// takes in the entire dataset
vector<HyperCircle> HyperCircle::createCircles(vector<Point> &dataset) {

    vector<HyperCircle> circles(dataset.size());

    // Parallel HC creation.
    #pragma omp parallel for
    for (int i = 0; i < dataset.size(); ++i) {
        Point &p = dataset[i];
        circles[i] = HyperCircle(0.0f, p.location, p.classification);
    }

    // update each circle's radius to our nearest neighbor.
    #pragma omp parallel for
    for (auto & circle : circles) {
        circle.findNearestNeighbor(dataset);
    }

    // delete entirely all those circles which had a radius of 0.0f. meaning their nearest neighbor is wrong class. 
    circles.erase(remove_if(circles.begin(), circles.end(),[](const HyperCircle& c) { return c.radius == 0.0f; }),circles.end());

    return circles;
}

// function which takes all our built circles, and starts deleting them as possible.
void HyperCircle::mergeCircles(vector<HyperCircle>& circles, vector<Point>& dataSet) {

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
            local.reserve(circles.size() / omp_get_num_threads());

            #pragma omp for
            for (int j = idx + 1; j < circles.size(); ++j) {

                // skip dead or wrong-class circles
                if (circles[j].centerPoint == nullptr || circles[j].classification != c.classification)
                    continue;

                // get our distance between our two circles. push that plus smaller guy radius.
                local.emplace_back(Utils::distance(c.centerPoint,circles[j].centerPoint, Point::numAttributes) + circles[j].radius, j);
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
            int circleID = smallestDistance.second;

            // if the distance is inside our existing radius, we can skip and mark this guy as eaten.
            if (newR2 <= c.radius) {
                mergable.push_back(circleID);
                continue;
            }

            atomic_int canMerge{1}; // shared flag
            #pragma omp parallel for schedule(static) shared(canMerge)
            for (auto pt : dataSet) {

                if (!canMerge)
                    continue;   // early-out after failure

                if (pt.classification == c.classification)
                    continue;

                if (Utils::distance(pt.location,c.centerPoint, Point::numAttributes) <= newR2) {
                    canMerge.store(0, memory_order_relaxed);
                    #ifdef _OPENMP
                    #pragma omp cancel for
                    #else
                    break;
                    #endif
                }
                #pragma omp cancellation point for
            }

            // if we can merge, update radius, and mark that this circle is toast.
            if (canMerge != 0) {
                c.radius = newR2;
                mergable.push_back(circleID);   // remember the real ID
            } else
                break;
        }

        // mark every circle we “ate” as null instead of erasing now
        for (int id : mergable)
            circles[id].centerPoint = nullptr;

    }

    // single compaction pass. remove all null centerpoints HC's
    circles.erase(remove_if(circles.begin(), circles.end(),[](const HyperCircle& hc){return hc.centerPoint == nullptr; }), circles.end());
}

bool HyperCircle::insideCircle(float *dataToCheck) {
    return (Utils::distance(centerPoint, dataToCheck, Point::numAttributes) <= radius);
}

// function which makes us a list of circles given some pre processed dataSet
vector<HyperCircle> HyperCircle::generateHyperCircles(vector<Point> &dataSet, int numClasses) {

    // generate our initial list of circles
    vector<HyperCircle> circles = createCircles(dataSet);

    cout << "Circles created...\nBeginning Merging." << endl;

    // merge our circles so that we can get larger circles
    mergeCircles(circles, dataSet);

    cout << "Circles merged...\nRemoving Circles" << endl;

    for (auto &circle : circles) {

        int insideCount = 0;

        // do a reduction to efficiently compute the size of the circle in terms of point count
        // we do the reduction because we are computing a ton of euclidean distances in the inside function.
        #pragma omp parallel for reduction(+ : insideCount)
        for (auto & point : dataSet) {

            // if this point is inside, increment numPoints
            if (circle.insideCircle(point.location)) {
                insideCount++;
            }
        }
        // update the value now that the for loop is over
        circle.numPoints = insideCount;
    }

    // remove circles which don't uniquely classify any points
    removeUselessCircles(circles, dataSet);

    cout << "Useless Circles Removed...\nWe generated:\t" << circles.size() << " circles." << endl;

    // get our count of how many circles per class
    numCirclesPerClass.resize(numClasses);
    for (auto &circle : circles)
        numCirclesPerClass[circle.classification]++;

    return circles;
}

// generates circles based on how big their radius can possible be of pure classification. then we simplify by removing useless circles.
vector<HyperCircle> HyperCircle::generateMaxDistanceBasedHyperCircles(vector<Point> &dataSet, int numClasses) {
    vector<HyperCircle> circles(dataSet.size());

    // Parallel HC creation.
    #pragma omp parallel for
    for (int i = 0; i < dataSet.size(); ++i) {
        Point &p = dataSet[i];
        circles[i] = HyperCircle(0.0f, p.location, p.classification);
    }

    #pragma omp parallel for
    for (auto & circle : circles) {
        circle.findMaxDistance(dataSet);
    }

    // count how many points are in each circle.
    for (auto &circle : circles) {

        int insideCount = 0;
        // do a reduction to efficiently compute the size of the circle in terms of point count
        // we do the reduction because we are computing a ton of euclidean distances in the inside function.
        #pragma omp parallel for reduction(+ : insideCount)
        for (auto & point : dataSet) {

            // if this point is inside, increment numPoints
            if (circle.insideCircle(point.location)) {
                insideCount++;
            }
        }
        // update the value now that the for loop is over
        circle.numPoints = insideCount;
    }

    // remove circles which don't uniquely classify any points
    removeUselessCircles(circles, dataSet);

    cout << "Useless Circles Removed...\nWe generated:\t" << circles.size() << " circles." << endl;

    // get our count of how many circles per class
    numCirclesPerClass.resize(numClasses);
    for (auto &circle : circles)
        numCirclesPerClass[circle.classification]++;

    return circles;
}

// simplification. removes circles which classify no points uniquely.
void HyperCircle::removeUselessCircles(vector<HyperCircle> &circles, vector<Point> &dataSet) {
    // vector to track how many points each circle had
    vector<int> circlePointCounts(circles.size(), 0);


    #pragma omp parallel
    {
        vector<int> localCounts(circles.size(), 0);   // private

        // take every point, and find it's biggest circle it fits inside of by radius
        #pragma omp for schedule(static)
        for (auto p : dataSet) {

            // get our biggest count, track it, get our point
            float biggestRadius = 0;
            int bestCircleIndex = -1;

            // find the best circle
            for (int circ = 0; circ < circles.size(); circ++) {
                auto &c = circles[circ];

                // we can do this because we always use PURE circles.
                if (c.classification != p.classification)
                    continue;


                if (c.insideCircle(p.location) && c.radius > biggestRadius) {
                    biggestRadius = c.numPoints;
                    bestCircleIndex = circ;
                }
            }
            if (bestCircleIndex != -1) {
                localCounts[bestCircleIndex]++;
            }
        }

        // merge private tallies into the shared array
        #pragma omp critical
        for (int i = 0; i < circlePointCounts.size(); ++i)
            circlePointCounts[i] += localCounts[i];

    }// end pragma

    // Erase circles that had no points assigned
    auto newEnd = remove_if(circles.begin(), circles.end(),
        [&, i = 0](const HyperCircle &c) mutable {
            return circlePointCounts[i++] == 0;
        });
    circles.erase(newEnd, circles.end());
}

int HyperCircle::classifyPoint(vector<HyperCircle> &circles, vector<Point> &train, float *dataToCheck, int classificationMode, int subMode,  int numClasses, int k = 5) {

    // here we use our different classification options.
    // first option is to just take whichever class we find our point in the most.
    // we could also use the average density of each circle, so number of points / area on average of each circle
    // another option is to use the total point count of each circle our point fell into
    int prediction = -1;
    switch (classificationMode) {

        case USE_CIRCLES: {

            vector<float> votes(numClasses, 0.0f);

            // smallest circles radius and class
            pair<float, int> smallestCircle {numeric_limits<float>::max(), -1};

            for (int i = 0; i < circles.size(); i++) {
                if (circles[i].insideCircle(dataToCheck)) {

                    // determine which style voting
                    switch (subMode) {

                        // regular vote, we just use the count of circles we're inside by class
                        case SIMPLE_MAJORITY: {
                            votes[circles[i].classification] += 1.0f;
                            break;
                        }

                        // vote with the amount of points in the circle.
                        case COUNT_VOTE: {
                            votes[circles[i].classification] += circles[i].numPoints;
                            break;
                        }


                        case DENSITY_VOTE: {
                            // simple count/radius
                            float r = max(circles[i].radius, 1e-6f);
                            float weight = circles[i].numPoints / r;
                            votes[circles[i].classification] += weight;
                            break;
                        }

                        case DISTANCE_VOTE: {
                            // count / distance from the center
                            float dist = Utils::distance(dataToCheck, circles[i].centerPoint, Point::numAttributes);
                            float weight = circles[i].numPoints / (dist + 1e-4f);
                            votes[circles[i].classification] += weight;
                            break;
                        }

                        case PER_CLASS_VOTE: {
                            // we add 1 / num circles of this class as a vote.
                            votes[circles[i].classification] += 1.0f / numCirclesPerClass[circles[i].classification];
                            break;
                        }

                        case SMALLEST_CIRCLE: {
                            // get our distance.
                            float distance = Utils::distance(circles[i].centerPoint, dataToCheck, Point::numAttributes);

                            // if we're inside, and this is smallest circle, we take this circle's classification.
                            if (distance < smallestCircle.first) {
                                smallestCircle.first = distance;
                                smallestCircle.second = i;
                            }
                            break;
                        }

                        // shut up the compiler
                        default: {
                            throw new runtime_error("Unknown classification mode");
                        }
                    } // voting switch
                } // inside
            } // circles loop

            // if we were looking for smallest circle, we can just return it from here.
            if (subMode == SMALLEST_CIRCLE) {
                return circles[smallestCircle.second].classification;
            }

            // start maxVotes at 0. that way if no votes are cast, we know that we have to classify with the fallback mechanisms
            float maxVotes = 0.0f;
            for (int cls = 0; cls < numClasses; ++cls) {
                if (votes[cls] > maxVotes) {
                    maxVotes = votes[cls];
                    prediction = cls;
                }
            }

            // end of using circles case.
            break;
        }

        // standard knn algorithm
        case REGULAR_KNN: {
            prediction = regularKNN(train, dataToCheck, k, numClasses);
            break;
        }

        // k nearest HC's by radius
        case K_NEAREST_CIRCLES: {
            prediction = kNearestCircle(circles, dataToCheck, k, numClasses);
            break;
        }

        // k nearest HC's by distance / radius. this way we know relatively how far outside a circles radius it was.
        case K_NEAREST_RATIOS: {
            prediction = kNearestCircleRatio(circles, dataToCheck, k, numClasses);
            break;
        }

        default:{}
    }

    // fallback mechanism
    return prediction;
}

int HyperCircle::regularKNN(vector<Point> &dataSet, float *point, int k, int numClasses) {

    vector<pair<float, int>> distances(dataSet.size());

    #pragma omp parallel for
    for (int dp = 0; dp < dataSet.size(); ++dp) {
        distances[dp] = {Utils::distance(dataSet[dp].location, point, Point::numAttributes), dataSet[dp].classification};
    }

    // sort up to kth element. clamping if needed
    if (k > distances.size())
        k = distances.size();

    nth_element(distances.begin(),distances.begin() + k, distances.end(),[](const auto& a, const auto& b){ return a.first < b.first; });

    // vote. weighting by the 1/distance.
    vector<float> votes(numClasses, 0.0f);
    for (int i = 0; i < k; ++i)
        votes[distances[i].second] += (1 / distances[i].first);

    // return our best class by finding max element
    return distance(votes.begin(),max_element(votes.begin(), votes.end()));
}

int HyperCircle::kNearestCircle(vector<HyperCircle> &circles, float *point, int k, int numClasses) {

    vector<pair<float, int>> distances(circles.size());

    #pragma omp parallel for
    for (int c = 0; c < circles.size(); ++c) {
        // save our distance and this circles class
        distances[c] = {Utils::distance(circles[c].centerPoint, point, Point::numAttributes), circles[c].classification};
    }

    nth_element(distances.begin(),distances.begin() + k, distances.end(),[](const auto& a, const auto& b){ return a.first < b.first; });

    // vote. weighting by the 1 / distance.
    vector<float> votes(numClasses, 0.0f);
    for (int i = 0; i < k; ++i)
        votes[distances[i].second] += (1 / distances[i].first);

    // return our best class by finding max element
    return distance(votes.begin(), max_element(votes.begin(), votes.end()));
}

int HyperCircle::kNearestCircleRatio(vector<HyperCircle> &circles, float *point, int k, int numClasses) {

    vector<pair<float, int>> distances(circles.size());

    #pragma omp parallel for
    for (int c = 0; c < circles.size(); ++c) {
        // save our distance / radius, and the class corresponding to this circle
        distances[c] = {(Utils::distance(circles[c].centerPoint, point, Point::numAttributes) / circles[c].radius), circles[c].classification};
    }

    nth_element(distances.begin(),distances.begin() + k, distances.end(),[](const auto& a, const auto& b){ return a.first < b.first; });

    // vote. weighting by the 1 / distance.
    vector<float> votes(numClasses, 0.0f);
    for (int i = 0; i < k; ++i)
        votes[distances[i].second] += (1 / distances[i].first);

    // return our best class by finding max element
    return distance(votes.begin(), max_element(votes.begin(), votes.end()));
}