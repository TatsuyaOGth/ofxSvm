#include "ofApp.h"

template<typename T>
vector<T> makeData(int label)
{
    vector<T> data;
    data.push_back(label);
    data.push_back(ofRandom(10));
    data.push_back(ofRandom(10));
    data.push_back(ofRandom(10));
    data.push_back(ofRandom(10));
    data.push_back(ofRandom(10));
    return data;
}

void ofApp::setup(){
    
    // make training data
    typedef float type;
    vector<vector<type> >train_dataset;
    train_dataset.push_back(makeData<type>(0));
    train_dataset.push_back(makeData<type>(0));
    train_dataset.push_back(makeData<type>(1));
    train_dataset.push_back(makeData<type>(1));
    ofxSvm::saveToFileDataset("train_dataset", train_dataset);
    
    
    // scaling training dataset
    ofxSvm::scaling("train_dataset", "train_dataset.scale", -1, 1, 0, 1);
    
    // train
    mSvm.train("train_dataset.scale", "model");
    mSvm.saveModelData("model");
    
    OF_EXIT_APP(0)
    
    // predict
    mSvm.predict("dataset", "model", "result");
    
    
    // test data
//    data.clear();
//    data.push_back(2.2);
//    data.push_back(42.6);
//    data.push_back(40.2);
//    data.push_back(10.2);
//    data.push_back(1.4);
//    mTestData.setData(0, data);
    
    // test
//    mSvm.loadModelData("model");
//    mSvm.classify(mTestData);
}

void ofApp::update(){

}

void ofApp::draw(){
    
    
}
