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

// tests the accuracy with our test set.
float testAccuracy(vector<HyperCircle> &circles, vector<Point> &testData) {

    vector<vector<int>> confusionMatrix(CLASS_MAP.size(), vector<int>(CLASS_MAP.size()));

    for (int p = 0; p < testData.size(); p++) {
        const auto &point = testData[p];
        const int predictedClass = HyperCircle::classifyPoint(circles, point.location, HyperCircle::SIMPLE_MAJORITY, NUM_CLASSES);

        if (predictedClass == -1)
            continue;

        // increment our count of predicted class for this point's real class. if they're same, that's good obviously.
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
    }

    // count how many we got right
    int totalRight = 0;
    for (int cls = 0; cls < confusionMatrix.size(); cls++) {
        totalRight += confusionMatrix[cls][cls];
    }

    // return our average
    return (float) totalRight / (float) testData.size();
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

        vector<HyperCircle> circles = HyperCircle::generateHyperCircles(trainingData);

        // get our accuracy on the test portion.
        totalAcc += testAccuracy(circles, testData);

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
                getline(cin, fileName);
                testData = readFile(fileName);

                Utils::waitForEnter();
                break;
            }
            // generates HC's from the training file
            case 3: {
                circles = HyperCircle::generateHyperCircles(trainData);
                cout << "Generated: " << circles.size() << " HyperCircles." << endl;
                Utils::waitForEnter();
                break;
            }
            // tests against a given test set
            case 4: {
                float acc = testAccuracy(circles, testData);
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
