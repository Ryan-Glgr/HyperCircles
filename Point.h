//
// Created by Ryan Gallagher on 5/27/25.
//

#ifndef POINT_H
#define POINT_H

class Point {
    public:
    // the point's attributes
    float *location;
    // the actual class
    int classification;

    // amount of attributes in dataset.
    static int numAttributes;

    Point(float *attributes, int cls) {
        location = attributes;
        classification = cls;
    }

};



#endif //POINT_H
