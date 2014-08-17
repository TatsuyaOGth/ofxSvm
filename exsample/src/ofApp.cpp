#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
    
    
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    test();
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}





void ofApp::test()
{
    vector<float> data;
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
    
    mSvm.scaling(-1, 1);
    mSvm.train();
}


