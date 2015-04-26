#include "ofApp.h"

void ofApp::setup(){
    
    vector<double> data;
    data.push_back(2.4);
    data.push_back(45.2);
    data.push_back(44.2);
    data.push_back(50.1);
    data.push_back(1.25);
    mSvm.setData(0, data);
    
    data.clear();
    data.push_back(3.3);
    data.push_back(40.1);
    data.push_back(41.2);
    data.push_back(50.2);
    data.push_back(2.32);
    mSvm.setData(0, data);
    
    data.clear();
    data.push_back(4.1);
    data.push_back(42.2);
    data.push_back(47.7);
    data.push_back(48.2);
    data.push_back(0.11);
    mSvm.setData(1, data);
    
    data.clear();
    data.push_back(2.4);
    data.push_back(38.2);
    data.push_back(40.1);
    data.push_back(54.3);
    data.push_back(2.25);
    mSvm.setData(1, data);
    
    cout << "\n\n----- set features data -----\n\n";
    mSvm.dumpDataset();
    mSvm.exportDataset("dataset");
    
    cout << "\n\n----- scaling -----\n\n";
    mSvm.scaling("dataset", "dataset.scale");
    mSvm.dumpDataset("dataset.scale");
    
    cout << "\n\n----- train -----\n\n";
    mSvm.train("dataset.scale", "model");
    
    cout << "\n\n----- predict -----\n\n";
    mSvm.predict("dataset", "model", "result");
}

void ofApp::update(){

}

void ofApp::draw(){
    
    
}
