#ifndef __ofxSvmExample__ofxSvm__
#define __ofxSvmExample__ofxSvm__

#include "ofMain.h"

enum OFXSVM_TYPE
{
    OFXSVM_TYPE_C_SVC = 0,
    OFXSVM_TYPE_NU_SVC = 1,
    OFXSVM_TYPE_ONE_CLASS_SVM = 2,
    OFXSVM_TYPE_EPSILON_SVR = 3,
    OFXSVM_TYPE_NU_SVR = 4
};

enum OFXSVM_KERNEL
{
    OFXSVM_KERNEL_LINER = 0,
    OFXSVM_KERNEL_POLYNOMIAL = 1,
    OFXSVM_KERNEL_RBF = 2,
    OFXSVM_KERNEL_SIGMOID = 3
};


//============================================
class ofxSvm
{
public:
    typedef multimap<int, vector<double> > dataset_type;
    
private:
    dataset_type mDataset;
    
    
    // training parameters
    
    OFXSVM_TYPE mSvmType;
    OFXSVM_KERNEL mKernelType;
    double   mDegree;
    double   mGamma;
    double   mCoef0;
    double   mCost;
    double   mNu;
    double   mEpsilon;
    double   mTolerance;
    double   mCachesize;
    bool     mUseShrinking;
    bool     mProbabilityEstimates;
    int      mCrossValidationNr;
    
public:
    ofxSvm():
    mSvmType(OFXSVM_TYPE_C_SVC),
    mKernelType(OFXSVM_KERNEL_RBF),
    mDegree(3),
    mGamma(1),
    mCoef0(0),
    mCost(1),
    mNu(0.5),
    mCachesize(100),
    mEpsilon(0.1),
    mTolerance(0.001),
    mUseShrinking(true),
    mProbabilityEstimates(false),
    mCrossValidationNr(0)
    {
    }
    
    void setData(const int lavel, vector<double>& features);
    
    void exportDataset(const string& filename);
    
    bool train(const string& dataset_file_name, const string& output_model_file_name);
    
    bool scaling(const string& data_file_name, const string& output_file_name,
                 const double x_lower = -1.0, const double x_upper = 1.0,
                 const double y_lower = 0, const double y_upper = 0);
    
    bool predict(const string& test_file_name, const string& model_file_name,
                 const string& result_file_name, const bool predict_probability = false);
    
    
    
    
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
    void setNumClossValidation(const int param) { mCrossValidationNr = param; }
    
    dataset_type& getDatasetRef() { return mDataset; }
    
    
    void dumpDataset();
    void dumpDataset(const string& data_file_name);
};

#endif /* defined(__ofxSvmExample__ofxSvm__) */
