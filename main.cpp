#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Point.h"
#include "HyperCircle.h"
#include "Utils.h"
#include <map>


using namespace std;

// used so that we can easily get the number of classes, names and whatnot
// very similar to github.com/austinsnyd3r/hyperblocks
map<string, int> CLASS_MAP;
int NUM_CLASSES = 0;
int Point::numAttributes = 0;
bool PRINTING = true;

vector<Point> readFile(const string &fileName) {

    vector<Point> data;
#ifdef _WIN32
    const string realName = "datasets\\" + fileName;
#else
    const string realName = "datasets/" + fileName;
#endif

    ifstream file(realName);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << fileName << endl;
        return data;
    }

    string line;

    // Read header to determine number of attributes
    if (!getline(file, line))
        return data;

    stringstream headerSS(line);
    int columnCount = 0;
    string headerToken;
    while (getline(headerSS, headerToken, ',')) {
        ++columnCount;
    }

    Point::numAttributes = columnCount - 1;

    // Parse each data line
    while (getline(file, line)) {
        stringstream ss(line);
        vector<string> tokens;
        string token;

        while (getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != columnCount) {
            cerr << "Skipping malformed row: " << line << endl;
            continue; // malformed row
        }

        float* attrs = new float[Point::numAttributes];
        try {
            for (int i = 0; i < Point::numAttributes; ++i) {
                attrs[i] = stof(tokens[i]);
            }
        } catch (...) {
            cerr << "Failed to parse floats on line: " << line << endl;
            delete[] attrs;
            continue; // skip row with invalid numeric data
        }

        // Handle class label
        string label = tokens.back();
        if (!CLASS_MAP.contains(label)) {
            CLASS_MAP[label] = NUM_CLASSES++;
        }
        int cls = CLASS_MAP[label];

        // throw our data into the vector
        data.emplace_back(attrs, cls);
    }

    return data;
}

// quick GPT made functions to load and save circles to and from a file.
static void saveCircles(const vector<HyperCircle>& circles, const string& filename) {
    ofstream out(filename, ios::binary);
    int32_t n = static_cast<int32_t>(circles.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& hc : circles) {
        // radius
        out.write(reinterpret_cast<const char*>(&hc.radius), sizeof(hc.radius));
        // classification
        out.write(reinterpret_cast<const char*>(&hc.classification), sizeof(hc.classification));
        // numPoints
        out.write(reinterpret_cast<const char*>(&hc.numPoints), sizeof(hc.numPoints));
        // centerPoint coordinates (Point::numAttributes floats)
        for (int d = 0; d < Point::numAttributes; ++d) {
            out.write(reinterpret_cast<const char*>(&hc.centerPoint[d]), sizeof(float));
        }
    }
    out.close();
}

// loads circles from a file
static vector<HyperCircle> loadCircles(const string& filename) {
    ifstream in(filename, ios::binary);
    int32_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));

    vector<HyperCircle> circles;
    circles.reserve(n);
    for (int32_t i = 0; i < n; ++i) {
        float   radius;
        int32_t cls;
        int32_t count;
        in.read(reinterpret_cast<char*>(&radius),    sizeof(radius));
        in.read(reinterpret_cast<char*>(&cls),       sizeof(cls));
        in.read(reinterpret_cast<char*>(&count),     sizeof(count));

        // allocate and read centerPoint array
        float* center = new float[Point::numAttributes];
        for (int d = 0; d < Point::numAttributes; ++d) {
            in.read(reinterpret_cast<char*>(&center[d]), sizeof(float));
        }

        HyperCircle hc(radius, center, cls);
        hc.numPoints = count;
        circles.push_back(hc);
    }
    in.close();
    return circles;
}

// tests the accuracy with our test set.
float testAccuracy(vector<HyperCircle> &circles, vector<Point> &train, vector<Point> &testData, int k) {

    vector<vector<int>> confusionMatrix(CLASS_MAP.size(), vector<int>(CLASS_MAP.size()));
    int unclassifiedCount = 0;
    for (int p = 0; p < testData.size(); ++p) {
        const auto &point = testData[p];

        int predictedClass = HyperCircle::classifyPoint(circles, train, point.location, HyperCircle::USE_CIRCLES, HyperCircle::SIMPLE_MAJORITY, NUM_CLASSES, k);

        if (predictedClass == -1) {
            // re‐classify with KNN and assign to predictedClass
            predictedClass = HyperCircle::classifyPoint(circles,train, point.location,HyperCircle::REGULAR_KNN, HyperCircle::PER_CLASS_VOTE, NUM_CLASSES, k);
            unclassifiedCount++;
        }

        // increment of the appropriate cell in confusionMatrix
        confusionMatrix[point.classification][predictedClass]++;
    }

    if (PRINTING) {
        cout << "CONFUSION MATRIX:" << endl;
        for (int cls = 0; cls < CLASS_MAP.size(); ++cls) {
            for (int row = 0; row < CLASS_MAP.size(); ++row) {
                cout << confusionMatrix[cls][row] << "\t|| ";
            }
            cout << endl;
        }
        cout << "UNCLASSIFIED BY THE HCs:\t" << unclassifiedCount << endl << endl;
    }

    // count how many we got right
    int totalRight = 0;
    for (int cls = 0; cls < confusionMatrix.size(); cls++) {
        totalRight += confusionMatrix[cls][cls];
    }

    // return our average
    return (float) totalRight / (float) testData.size();
}

// finds best HC voting style
void findBestHCVoting (vector<HyperCircle> &circles, vector<Point> &train, vector<Point> &testData) {

    // We'll store accuracy for each submode in this array (indices correspond to enum values 0–4).
    float accuracies[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Loop over each HC voting submode
    for (int subMode = HyperCircle::SIMPLE_MAJORITY; subMode <= HyperCircle::SMALLEST_CIRCLE; ++subMode) {

        vector<vector<int>> confusionMatrix(CLASS_MAP.size(), vector<int>(CLASS_MAP.size()));
        int unclassifiedCount = 0;

        int k = 5;

        for (int p = 0; p < testData.size(); ++p) {
            const auto &point = testData[p];

            int predictedClass = HyperCircle::classifyPoint(circles,train,point.location, HyperCircle::USE_CIRCLES, subMode, NUM_CLASSES, k);

            // if not covered by our circle, use KNN.
            if (predictedClass == -1) {
                predictedClass = HyperCircle::classifyPoint(circles,train,point.location,HyperCircle::REGULAR_KNN,HyperCircle::PER_CLASS_VOTE, NUM_CLASSES, k);
                unclassifiedCount++;
            }

            // increment of the appropriate cell in confusionMatrix
            confusionMatrix[point.classification][predictedClass]++;
        }

        if (PRINTING) {
            // Print header for this submode
            switch (subMode) {
                case HyperCircle::SIMPLE_MAJORITY:
                    cout << "=== CONFUSION MATRIX (SIMPLE_MAJORITY) ===" << endl;
                    break;
                case HyperCircle::COUNT_VOTE:
                    cout << "=== CONFUSION MATRIX (COUNT_VOTE) ===" << endl;
                    break;
                case HyperCircle::DENSITY_VOTE:
                    cout << "=== CONFUSION MATRIX (DENSITY_VOTE) ===" << endl;
                    break;
                case HyperCircle::DISTANCE_VOTE:
                    cout << "=== CONFUSION MATRIX (DISTANCE_VOTE) ===" << endl;
                    break;
                case HyperCircle::PER_CLASS_VOTE:
                    cout << "=== CONFUSION MATRIX (PER_CLASS_VOTE) ===" << endl;
                    break;
                case HyperCircle::SMALLEST_CIRCLE:
                    cout << "=== CONFUSION MATRIX (SMALLEST CIRCLE) ===" << endl;
                    break;
            }

            for (int cls = 0; cls < CLASS_MAP.size(); ++cls) {
                for (int row = 0; row < CLASS_MAP.size(); ++row) {
                    cout << confusionMatrix[cls][row] << "\t|| ";
                }
                cout << endl;
            }
            cout << "UNCLASSIFIED BY THE HCs:\t" << unclassifiedCount << endl << endl;
        }

        // count how many we got right
        int totalRight = 0;
        for (int cls = 0; cls < confusionMatrix.size(); cls++) {
            totalRight += confusionMatrix[cls][cls];
        }

        // compute accuracy for this submode
        accuracies[subMode] = (float) totalRight / (float) testData.size();
    }

    // After testing all submodes, print out each accuracy
    if (PRINTING) {
        cout << "=== HC Voting Submode Accuracies ===" << endl;
        cout << "SIMPLE_MAJORITY: " << accuracies[HyperCircle::SIMPLE_MAJORITY] << endl;
        cout << "COUNT_VOTE:      " << accuracies[HyperCircle::COUNT_VOTE] << endl;
        cout << "DENSITY_VOTE:    " << accuracies[HyperCircle::DENSITY_VOTE] << endl;
        cout << "DISTANCE_VOTE:   " << accuracies[HyperCircle::DISTANCE_VOTE] << endl;
        cout << "PER_CLASS_VOTE:  " << accuracies[HyperCircle::PER_CLASS_VOTE] << endl;
        cout << "SMALLEST_CIRCLE:" << accuracies[HyperCircle::SMALLEST_CIRCLE] << endl;
        cout << endl;
    }
}

void findBestKNNStyle(vector<HyperCircle> &circles, vector<Point> &train, vector<Point> &testData) {

    // different k values to test
    vector<float> kVals {1, 3, 5, 7, 9, 13, 15, 21, 25};

    vector<vector<int>> confusionMatrix(CLASS_MAP.size(), vector<int>(CLASS_MAP.size()));
    int unclassifiedCount = 0;

    vector<Point> pointsNotClassified;

    // classify all points with HC's as normal
    for (int p = 0; p < testData.size(); ++p) {
        const auto &point = testData[p];

        int predictedClass = HyperCircle::classifyPoint(circles,train,point.location, HyperCircle::USE_CIRCLES, HyperCircle::SIMPLE_MAJORITY, NUM_CLASSES, -1);

        // if not covered by our circle, use KNN.
        if (predictedClass == -1) {
            unclassifiedCount++;
            // push back this point so that we know we have to run KNN on it.
            pointsNotClassified.push_back(point);
        }

        // Atomic increment of the appropriate cell in confusionMatrix
        confusionMatrix[point.classification][predictedClass]++;
    }
    cout << "HC's Covered: " << unclassifiedCount << " of the test points!" << endl;

    for (int k = 0; k < kVals.size(); ++k) {

        // Loop over each HC voting submode
        float accuracies[3] = {0.0f, 0.0f, 0.0f};

        for (int subMode = HyperCircle::REGULAR_KNN; subMode <= HyperCircle::K_NEAREST_RATIOS; ++subMode) {

            // copy the confusion matrix which was generated by the HC's
            auto thisConfigConfusionMatrix = confusionMatrix;

            for (int p = 0; p < pointsNotClassified.size(); ++p) {
                const auto &point = pointsNotClassified[p];

                int predictedClass = HyperCircle::classifyPoint(circles,train,point.location, subMode, -1, NUM_CLASSES, kVals[k]);

                // if not covered by our circle, use KNN.
                if (predictedClass == -1) {
                    cout << "KNN RETURNED -1!" << endl;
                }

                // increment of the appropriate cell in confusionMatrix
                thisConfigConfusionMatrix[point.classification][predictedClass]++;
            }

            if (PRINTING) {
                cout << "K = " << k << endl;
                // Print header for this submode
                switch (subMode) {
                    case HyperCircle::REGULAR_KNN:
                        cout << "=== CONFUSION MATRIX (REGULAR KNN) ===" << endl;
                    break;
                    case HyperCircle::K_NEAREST_CIRCLES:
                        cout << "=== CONFUSION MATRIX (NEAREST CIRCLE) ===" << endl;
                    break;
                    case HyperCircle::K_NEAREST_RATIOS:
                        cout << "=== CONFUSION MATRIX (NEAREST RATIOS) ===" << endl;
                    break;
                }

                for (int cls = 0; cls < CLASS_MAP.size(); ++cls) {
                    for (int row = 0; row < CLASS_MAP.size(); ++row) {
                        cout << thisConfigConfusionMatrix[cls][row] << "\t|| ";
                    }
                    cout << endl;
                }
            }

            // count how many we got right
            int totalRight = 0;
            for (int cls = 0; cls < confusionMatrix.size(); cls++) {
                totalRight += thisConfigConfusionMatrix[cls][cls];
            }

            // compute accuracy for this submode
            accuracies[subMode - HyperCircle::REGULAR_KNN] = (float) totalRight / (float) testData.size();

        } // end submode

        // After testing all submodes, print out each accuracy
        if (PRINTING) {
            cout << "=== KNN Voting Submode Accuracies with K value: " << kVals[k] << "===" << endl;
            cout << "REGULAR_KNN: " << accuracies[0] << endl;
            cout << "K_NEAREST_CIRCLES: " << accuracies[1] << endl;
            cout << "K_NEAREST_RATIOS: " << accuracies[2] << endl;
            cout << endl;
        }
    } // end k val

}

// return is accuracy, then average circle count
pair<float, float> kFoldValidation(int numFolds, vector<Point> &allData) {

    // first we use our util function to split up all our data into different training and testing folds.
    vector<vector<Point>> kBuckets = Utils::stratifiedKFolds(numFolds, allData);

    // now, we run our loop numFolds times. gathering info each time.
    float totalAcc = 0.0f;
    int totalCircles = 0;
    for (int fold = 0; fold < numFolds; ++fold) {

        // setting up train and test split for this iteration
        vector<Point> trainingData;
        vector<Point> testData;
        for (int trainFold = 0; trainFold < numFolds; ++trainFold) {
            if (trainFold == fold) {
                testData = kBuckets[fold];
            }
            else
                trainingData.insert(trainingData.end(), kBuckets[trainFold].begin(), kBuckets[trainFold].end());
        }

        vector<HyperCircle> circles = HyperCircle::generateHyperCircles(trainingData, NUM_CLASSES);

        // get our accuracy on the test portion.
        int k = 5;
        totalAcc += testAccuracy(circles, trainingData, testData, k);

        // add our count so we can track how many circles we needed.
        totalCircles += circles.size();
    }

    float avgAcc = totalAcc / (float) numFolds;
    float avgCircles = totalCircles / (float) numFolds;

    if (PRINTING) {
        printf("%d FOLD CROSS VALIDATION ACCURACY: %.3f", numFolds, avgAcc);
        printf("AVERAGE NUMBER OF HYPERCIRCLES NEEDED:\t%.2f", avgCircles);
    }

    return {avgAcc, avgCircles};
}

// frees all the points
void cleanupPoints(vector<Point> &data) {
    for (int i = 0; i < data.size(); ++i) {
        delete[] data[i].location;
    }
}

int main() {

    int choice;
    vector<Point> trainData;
    vector<Point> testData;
    vector<HyperCircle> circles;
    bool running = true;
    while (running) {

        // print main menu and get input
        Utils::displayMainMenu();
        cin >> choice;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        switch (choice) {

            // import training data
            case 1: {
                // get our user input for the filename.
                cout << "Enter training data filename: " << endl;
                #ifdef _WIN32
                system("dir datasets/");
                #else
                system("ls datasets/");
                #endif

                string fileName;
                getline(cin >> ws, fileName); // eat leading whitespace

                // reset our global variables
                CLASS_MAP.clear();
                NUM_CLASSES = 0;

                // get our Points
                trainData = readFile(fileName);

                cout << "FOUND: " << trainData.size() << " points in that file." << endl;
                Utils::waitForEnter();
                break;
            }
            // get user input for test file.
            case 2: {
                cout << "Enter testing data filename: " << endl;
                string fileName;

                #ifdef _WIN32
                system("dir datasets/");
                #else
                system("ls datasets/");
                #endif

                getline(cin, fileName);
                testData = readFile(fileName);

                Utils::waitForEnter();
                break;
            }
            // generates HC's from the training file
            case 3: {
                circles = HyperCircle::generateHyperCircles(trainData, NUM_CLASSES);
                cout << "Generated: " << circles.size() << " HyperCircles." << endl;
                Utils::waitForEnter();
                break;
            }
            // tests against a given test set
            case 4: {
                int k = 5;
                float acc = testAccuracy(circles, trainData,testData, k);
                cout << "Accuracy: " << acc << endl;
                Utils::waitForEnter();
                break;
            }
            // k fold validation
            case 5: {

                int numFolds;
                cout << "How many folds of cross validation (K value) ?" << endl;
                cin >> numFolds;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');

                kFoldValidation(numFolds, trainData);

                Utils::waitForEnter();
                break;
            }

            case 6: {
                cout << "Enter HC's file name to save to: " << endl;
                string fileName;
                getline(cin, fileName);

                saveCircles(circles, fileName);

                Utils::waitForEnter();
                break;
            }

            case 7: {
                cout << "Enter HC's file name to load from: " << endl;
                string fileName;
                getline(cin, fileName);

                circles = loadCircles(fileName);
                cout << "Loaded: " << circles.size() << " points in that file." << endl;

                Utils::waitForEnter();
                break;
            }

            case 8: {
                findBestHCVoting(circles, trainData, testData);
                Utils::waitForEnter();
                break;
            }

            case 9: {
                findBestKNNStyle(circles, trainData, testData);
                Utils::waitForEnter();
                break;
            }
            case -1: {
                running = false;
                break;
            }
            default: {
                break;
            }
        }
    }

    // delete our dynamically allocated memory
    cleanupPoints(trainData);
    cleanupPoints(testData);

    return 0;
}
