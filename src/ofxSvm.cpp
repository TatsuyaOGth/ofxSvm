#include "ofxSvm.h"

#ifdef __APPLE__
static const string SEPARATOR = "/";
static const string CD_BIN = "../../../../../../../addons/ofxSvm/bin"; //TODO: not elegant...
#elif defined __WIN32__
static const string SEPARATOR = "\";
static const string CD_BIN = "??????? \bin"; //TODO: i dont know original path in case win32
#endif

static const string OFXSVM_LOG_TITLE = "ofxSvm";

int ofxSvm::commandAtLibsDir(const string &cmd)
{
    return system(("cd " + CD_BIN + ";" + cmd).c_str());
}

void ofxSvm::outputTrainData()
{
    if (mDataSet.empty()) {
        ofLogError(OFXSVM_LOG_TITLE) << "dataset is empty";
        return false;
    }
    
    ofstream out((CD_BIN + "/data.dat").c_str());
    for (DATASET_TYPE::iterator it = mDataSet.begin(); it != mDataSet.end(); it++) {
        out << it->first;
        for (int i = 0; i < it->second.size(); i++) {
            out << " " << (i + 1) << ":" << it->second[i];
        }
        out << endl;
    }
    out.close();
}

void ofxSvm::reset()
{
    mDataSet.clear();
    mDidScaling = false;
}

void ofxSvm::test()
{
    {
        vector<string> datas;
        
        datas.push_back("+0 1:2.4 2:45.2 3:44.2 4:55.1 5:1.12");
        datas.push_back("+0 1:3.3 2:40.1 3:41.2 4:50.2 5:2.32");
        datas.push_back("+2 1:4.1 2:42.2 3:47.7 4:48.2 5:0.11");
        datas.push_back("+1 1:2.4 2:38.2 3:40.1 4:54.3 5:2.25");
        
        
        ofstream out((CD_BIN + "/data.dat").c_str());
        for (vector<string>::iterator it = datas.begin(); it != datas.end(); it++) out << *it << endl;
        out.close();
        
        int ret = 0;
        ret = commandAtLibsDir("./svm-scale -u 1 -l 0 data.dat > data.scale");
        ofLogNotice() << ret;
        
        ret = commandAtLibsDir("./svm-train data.scale model > train.");
        ofLogNotice() << ret;
        
    }
    
    //-------------------------------------------
    
    {
        vector<string> inputdata;
        
        inputdata.push_back("+1 1:2.4 2:45.2 3:44.2 4:55.1 5:1.12");
//        inputdata.push_back("+0 1:3.3 2:40.1 3:41.2 4:50.2 5:2.32");
//        inputdata.push_back("+1 1:4.1 2:42.2 3:47.7 4:48.2 5:0.11");
//        inputdata.push_back("+1 1:2.4 2:38.2 3:40.1 4:54.3 5:2.25");
        
        ofstream out((CD_BIN + "/input.dat").c_str());
        for (vector<string>::iterator it = inputdata.begin(); it != inputdata.end(); it++) out << *it << endl;
        out.close();
        
        int ret = 0;
        ret = commandAtLibsDir("./svm-predict input.dat model output > predict.out");
        ofLogNotice() << ret;
    }
}

bool ofxSvm::scaling(const float min, const float max)
{
    mScaleMin = min;
    mScaleMax = max;
    scaling();
}

bool ofxSvm::scaling()
{
    outputTrainData();
    
    string minStr = ofToString(mScaleMin);
    string maxStr = ofToString(mScaleMax);
    int ret = commandAtLibsDir("./svm-scale -u " + maxStr + " -l " + minStr + " data.dat > data.dat.scale");
    mDidScaling = true;
    return (ret == 0 ? true : false);
}

bool ofxSvm::train()
{
    if (!mDidScaling) {
        outputTrainData();
    }
    
    // set parameters
    DATASET_TYPE::iterator it = mDataSet.begin();
    const int k = it->second.size() - 1;
    stringstream ss;
    ss << " -s " << static_cast<int>(mSvmType);
    ss << " -t " << static_cast<int>(mKernelType);
    ss << " -d " << mDegree;
    ss << " -g " << mGamma / k;
    ss << " -r " << mCoef0;
    ss << " -c " << mCost;
    ss << " -n " << mNu;
    ss << " -m " << mCachesize;
    ss << " -e " << mEpsilon;
    ss << " -v " << mN;
    
    // traning
    string dataFile;
    mDidScaling ? dataFile = " data.dat.scale" : dataFile = " data.dat";
    string command = "./svm-train" + ss.str() + dataFile + " model";
    int ret = commandAtLibsDir(command);
    
    // reset
    reset();
    
    return (ret == 0 ? true : false);
}

bool ofxSvm::setData(const int dataClass, vector<float> & data)
{
    bool ret = true;
    if (!mDataSet.empty()) {
        int inSize = data.size();
        for (DATASET_TYPE::iterator it = mDataSet.begin(); it != mDataSet.end(); it++) {
            ret = inSize == it->second.size();
        }
    }
    if (!ret) ofLogWarning(OFXSVM_LOG_TITLE) << "in different size";
    
    mDataSet.insert(make_pair(dataClass, data));
    
    return ret;
}




