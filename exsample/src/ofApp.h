#pragma once

#include "ofMain.h"
#include "ofxSvm.h"

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    ofxSvm mSvm;
    ofxSvmData mTrainData, mTestData;
};
