#include "ofApp.h"

void ofApp::setup(){
    
    // training data
    vector<double> data;
    data.push_back(2.4);
    data.push_back(45.2);
    data.push_back(44.2);
    data.push_back(50.1);
    data.push_back(1.25);
    mTrainData.setData(0, data);
    
    data.clear();
    data.push_back(3.3);
    data.push_back(40.1);
    data.push_back(41.2);
    data.push_back(50.2);
    data.push_back(2.32);
    mTrainData.setData(0, data);
    
    data.clear();
    data.push_back(4.1);
    data.push_back(42.2);
    data.push_back(47.7);
    data.push_back(48.2);
    data.push_back(0.11);
    mTrainData.setData(1, data);
    
    data.clear();
    data.push_back(2.4);
    data.push_back(38.2);
    data.push_back(40.1);
    data.push_back(54.3);
    data.push_back(2.25);
    mTrainData.setData(1, data);
    
    mTrainData.exportDataset("dataset");
    
    // scaling
//    ofxSvmData::scaling("dataset", "dataset.scale", -1, 1, -1, 1);
    
    // train
    mSvm.train("dataset", "model");
    mSvm.saveModelData("model2");
    
    // predict
    mSvm.predict("dataset", "model", "result");
    
    
    
    // test data
    data.clear();
    data.push_back(2.2);
    data.push_back(42.6);
    data.push_back(40.2);
    data.push_back(10.2);
    data.push_back(1.4);
    mTestData.setData(0, data);
    
    // test
    mSvm.loadModelData("model");
    mSvm.classify(mTestData);
}

void ofApp::update(){

}

void ofApp::draw(){
    
    
}
