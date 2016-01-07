#include "ofApp.h"

double rand_normal( double mu, double sigma );

void ofApp::setup(){
    
//    ofSetLogLevel(OF_LOG_VERBOSE);
    ofBackground(60);
    
    // make sample data
    //--------------------------------------------------------
    mSamples.resize(3);
    mCurrentLabel = 0;
    
    for (int i = 0; i < 10; ++i)
        mSamples[0].push_back(ofVec2f(rand_normal(0.5, 0.15) * ofGetWidth(),
                                      rand_normal(0.3, 0.15) * ofGetHeight() - 150)); //<----- 150 is margin for bottom of the window
    for (int i = 0; i < 10; ++i)
        mSamples[1].push_back(ofVec2f(rand_normal(0.3, 0.15) * ofGetWidth(),
                                      rand_normal(0.7, 0.15) * ofGetHeight() - 150));
    for (int i = 0; i < 10; ++i)
        mSamples[2].push_back(ofVec2f(rand_normal(0.7, 0.15) * ofGetWidth(),
                                      rand_normal(0.7, 0.15) * ofGetHeight() - 150));
    
    
    // initialize predicted pixels
    //--------------------------------------------------------
    mPredictedPanel.allocate(ofGetWidth(), ofGetHeight() - 150, OF_IMAGE_COLOR);
    unsigned char* pix = mPredictedPanel.getPixels();
    const int size = mPredictedPanel.getWidth() * mPredictedPanel.getHeight() * 3;
    for (int i = 0; i < size; ++i)
    {
        pix[i] = 99; // default color
    }
    mPredictedPanel.update();
    
    
    // setup gui panel
    //--------------------------------------------------------
    mGui.setup("PARAMETERS");
    mGui.add(mGamma.set("GAMMA", 0.1, 0.0, 1));
    mGui.add(mCost.set("COST", 1, 0, 10));
    mGui.setPosition(10, 580);
}



void ofApp::draw(){
    
    ofBackground(30);
    ofSetColor(255);
    
    // draw predicted pixels
    //--------------------------------------------------------
    mPredictedPanel.draw(0, 0);
    
    
    
    // draw sample data
    //--------------------------------------------------------
    ofFill();
    for (int i = 0; i < mSamples.size(); ++i)
    {
        ofSetColor(ofColor::fromHsb(i * (255 / 3), 255, 255));
        for (int j = 0; j < mSamples[i].size(); ++j)
        {
            ofCircle(mSamples[i][j], 4);
        }
    }
    
    
    // draw explanation text
    //--------------------------------------------------------
    stringstream ss;
    ss << "mouse click: put new vector" << endl;
    ss << "1/2/3 key: change label" << endl;
    ss << "c key: crear all vectors" << endl;
    ss << "space bar: SVM train and show result of predict";
    ofSetColor(255, 255, 255);
    ofDrawBitmapString(ss.str(), 10, 520);
    
    
    
    // draw gui panel
    //--------------------------------------------------------
    mGui.draw();
    
    
    // draw cursor
    //--------------------------------------------------------
    if (ofGetMouseY() < ofGetHeight() - 150)
    {
        ofSetColor(ofColor::fromHsb(mCurrentLabel * (255 / 3), 255, 255));
        ofNoFill();
        ofSetLineWidth(2);
        ofCircle(ofGetMouseX(), ofGetMouseY(), 5);
    }
}



void ofApp::keyPressed(int key){
    
    switch (key){
        case '1': mCurrentLabel = 0; break;
        case '2': mCurrentLabel = 1; break;
        case '3': mCurrentLabel = 2; break;
        case 'c': for (auto& e : mSamples) e.clear(); break;
        case ' ': svm_execute(); break;
        case 's': mSvm.saveModel("model.dat"); break;
        case 'l': mSvm.loadModel("model.dat"); break;
    }
}



void ofApp::mousePressed(int x, int y, int button){
    if (y < 500)
    {
        mSamples[mCurrentLabel].push_back(ofVec2f(x, y));
    }
}



void ofApp::svm_execute(){
    
    // reset
    mTrainData.clear();
    mSvm.clear();
    
    
    // add train data
    //--------------------------------------------------------
    for (int i = 0; i < mSamples.size(); ++i)
    {
        for (int j = 0; j < mSamples[i].size(); ++j)
        {
            vector<double> vec;
            vec.push_back(mSamples[i][j].x);
            vec.push_back(mSamples[i][j].y);
            mTrainData.add(i + 1, vec);
        }
    }
    
    
    // scaling
    //--------------------------------------------------------
    //mTrainData.scale(0.0, 1.0);
    
    
    
    // set train parameters
    //--------------------------------------------------------
    mSvm.setSvmType(C_SVC);
    mSvm.setKernelType(LINEAR);
    mSvm.setCost(mCost);
    mSvm.setGamma(mGamma);
    mSvm.setCoef0(0);
    mSvm.setCachssize(100);
    mSvm.setEpsilon(1e-3);
    mSvm.setShrinking(true);
    mSvm.setProbability(false);
    mSvm.setDegree(3);
    mSvm.setNu(0.5);
    mSvm.setP(0.1);
    mSvm.setNrWeight(0);
    mSvm.setWeightLabel(NULL);
    mSvm.setWeight(NULL);
    
    
    
    // do train
    //--------------------------------------------------------
    mSvm.train(mTrainData);
    
    
    
    // predict
    //--------------------------------------------------------
    mTestData.clear();
    ofLogNotice() << "predict training samples ...";
    ofLogNotice() << "(target label) => (predictive result)";
    
    int correct_count = 0;
    int wrong_count = 0;
    for (int i = 0; i < mSamples.size(); ++i)
    {
        for (int j = 0; j < mSamples[i].size(); ++j)
        {
            vector<double> testvec;
            testvec.push_back(mSamples[i][j].x);
            testvec.push_back(mSamples[i][j].y);
            mTestData.add(0, testvec);
        }
    }
    
    //mTestData.scale(-1, 1);
    vector<double> predictResults = mSvm.predict(mTestData);
    
    vector<double>::iterator predictResIt = predictResults.begin();
    for (int i = 0; i < mSamples.size(); ++i)
    {
        for (int j = 0; j < mSamples[i].size(); ++j)
        {
            ofLogNotice() << (i + 1) << " => " << *predictResIt;
            if (*predictResIt == (i + 1))
                ++correct_count;
            else
                ++wrong_count;
            ++predictResIt;
        }
    }

    ofLogNotice() << "done";
    ofLogNotice() << "RESULT : correct=" << correct_count << " : wrong=" << wrong_count;
    ofLogNotice() << "Accuracy[%]=" << (static_cast<double>(correct_count) / static_cast<double>(correct_count + wrong_count) * 100.0);
    
    
    
    // update background color by predictive results
    //--------------------------------------------------------
    mTestData.clear();
    const int w = mPredictedPanel.getWidth();
    const int h = mPredictedPanel.getHeight();
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            vector<double> testvec;
            testvec.push_back(x);
            testvec.push_back(y);
            mTestData.add(0, testvec);
        }
    }
    
    predictResults.clear();
//    mTestData.scale(-1, 1);
    predictResults = mSvm.predict(mTestData);
    assert(predictResults.size() == w * h);
    
    unsigned char* bgPix = mPredictedPanel.getPixels();
    int ch = mPredictedPanel.getPixelsRef().getNumChannels();
    for (int i = 0; i < predictResults.size(); ++i)
    {
        ofColor col = ofColor::fromHsb((predictResults[i] - 1) * (255 / mSamples.size()), 255, 99);
        bgPix[i * ch + 0] = col.r;
        bgPix[i * ch + 1] = col.g;
        bgPix[i * ch + 2] = col.b;
    }
    mPredictedPanel.update();
}



double rand_normal( double mu, double sigma )
{
    double z = sqrt( -2.0 * log(ofRandomuf()) ) * sin( 2.0 * M_PI * ofRandomuf() );
    return ofClamp(mu + sigma * z, 0, 1);
}
