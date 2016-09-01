#pragma once

#include "ofMain.h"
#include "ofxSvm.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp{
public:
    void setup();
    void draw();
    void keyPressed( int key );
    void mousePressed( int x, int y, int button );
    void svm_execute();
    
    ofxSvm mSvm;
    ofxSvm::Data mTrainData;
    ofxSvm::Data mTestData;
    
    vector<vector<ofVec2f> > mSamples; // label, vector[]
    vector<int> mSupportVectors;
    int mCurrentLabel;
    ofImage mPredictedPanel;
    
    ofxPanel mGui;
    ofParameter<float> mGamma;
    ofParameter<float> mCost;
};
