#ifndef __ofxSvmExample__ofxSvm__
#define __ofxSvmExample__ofxSvm__

#include "ofMain.h"

enum OFXSVM_TYPE
{
    OFXSVM_TYPE_C_SVC = 0,          ///< [default]
    OFXSVM_TYPE_NU_SVC = 1,
    OFXSVM_TYPE_ONE_CLASS_SVM = 2,
    OFXSVM_TYPE_EPSILON_SVR = 3,
    OFXSVM_TYPE_NU_SVR = 4
};

enum OFXSVM_KERNEL
{
    OFXSVM_KERNEL_LINER = 0,
    OFXSVM_KERNEL_POLYNOMIAL = 1,
    OFXSVM_KERNEL_RBF = 2,          ///< [default]
    OFXSVM_KERNEL_SIGMOID = 3
};

class ofxSvm
{
    typedef multimap<int, vector<float> > DATASET_TYPE;
    
    // data set
    DATASET_TYPE mDataSet;
    
    // scaling parames
    float   mScaleMin;
    float   mScaleMax;
    bool    mDidScaling;
    
    // traing params
    OFXSVM_TYPE mSvmType;
    OFXSVM_KERNEL mKernelType;
    float   mDegree;
    float   mGamma;
    float   mCoef0;
    float   mCost;
    float   mNu;
    float   mEpsilon;
    float   mCachesize;
    bool    mUseShrinking;
    int     mN;
    
    int commandAtLibsDir(const string & cmd);
    void outputTrainData();
    void reset();
    
public:
    
    ofxSvm():
    mScaleMin(-1),
    mScaleMax( 1),
    mSvmType(OFXSVM_TYPE_C_SVC),
    mKernelType(OFXSVM_KERNEL_RBF),
    mDegree(3),
    mGamma(1),
    mCoef0(0),
    mCost(1),
    mNu(0.5),
    mCachesize(100),
    mEpsilon(0.001),
    mUseShrinking(true),
    mN(2)
    {
    }
    
    void test();
    
    bool setData(const int dataClass, vector<float> & data);
    bool scaling();
    bool scaling(const float min, const float max);
    bool train();
    
    void setScaleMin(const float v) { mScaleMin = v; }
    void setScaleMax(const float v) { mScaleMax = v; }
    void setSvmType(const OFXSVM_TYPE type) { mSvmType = type; }
    void setKernelType(const OFXSVM_KERNEL type) { mKernelType = type; }
    void setDegree(const float param) { mDegree = param; }
    void setGamma(const float param) { mGamma = param; }
    void setCoef0(const float param) { mCoef0 = param; }
    void setCost(const float param) { mCost = param; }
    void setNu(const float param) { mNu = param; }
    void setCachssize(const float param) { mCachesize = param; }
    void setEpsilon(const float param) { mEpsilon = param; }
    void setUseShrinking(const bool param) { mUseShrinking = param; }
    void setNumClossValidation(const int param) { mN = param; }
    
};

#endif /* defined(__ofxSvmExample__ofxSvm__) */
