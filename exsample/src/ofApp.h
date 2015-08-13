#pragma once

#include "ofMain.h"
#include "ofxSvm.h"

class ofApp : public ofBaseApp{
public:
    void setup();
    void draw();
    void keyPressed( int key );
    void mousePressed( int x, int y, int button );
    void svm_execute();
    
    ofxSvm mSvm;
    
    vector<vector<ofVec2f> > mSamples; // label, vector[]
    vector<int> mSupportVectors;
    int mCurrentLabel;
    ofImage mPredictedPanel;
};
