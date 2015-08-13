#include "ofxSvm.h"

#define LOG_MODULE "ofxSvm::"+string(__FUNCTION__)

ofxSvm::ofxSvm() : mModel(NULL), mDimension(0)
{
    defaultParams();
}

ofxSvm::~ofxSvm()
{
    svm_destroy_param(&mParam);
    svm_free_and_destroy_model(&mModel);
}

void ofxSvm::defaultParams()
{
    mParam.svm_type     = C_SVC;
    mParam.kernel_type  = RBF;
    mParam.degree       = 3;
    mParam.gamma        = 0; // 1/n
    mParam.coef0        = 0;
    mParam.C            = 1;
    mParam.nu           = 0.5;
    mParam.cache_size   = 100;
    mParam.eps          = 0.1;
    mParam.p            = 0.001;
    mParam.shrinking    = 1;
    mParam.probability  = 0;
    mParam.weight_label = NULL;
    mParam.weight       = NULL;
}

void ofxSvm::checkDimension(int length)
{
    if (mDimension > 0 && mDimension != length)
    {
        ofLogWarning(LOG_MODULE, "got different dimensions, set data");
        mData.clear();
    }
}

int ofxSvm::addData(int label, vector<double>& vec)
{
    checkDimension(vec.size());
    
    mData.insert(make_pair(label, vec));
    mDimension = vec.size();
    
    if (ofGetLogLevel() == OF_LOG_VERBOSE)
    {
        stringstream ss;
        for (const auto v : vec) ss << v << " ";
        ss << "EOS";
        ofLogVerbose(LOG_MODULE, "add data, label: " + ofToString(label) + " vec: " + ss.str());
    }
    
    return mData.size();
}

int ofxSvm::addData(int label, double *vec, int length)
{
    checkDimension(length);
    
    vector<double> v;
    for (int i = 0; i < length; ++i)
    {
        v.push_back(vec[i]);
    }
    mData.insert(make_pair(label, v));
    mDimension = length;
    
    if (ofGetLogLevel() == OF_LOG_VERBOSE)
    {
        stringstream ss;
        for (int i = 0; i < length; ++i) ss << vec[i] << " ";
        ss << "EOS";
        ofLogVerbose(LOG_MODULE, "add data, label: " + ofToString(label) + " vec: " + ss.str());
    }
    
    return mData.size();
}

void ofxSvm::creatData()
{
    mData.clear();
}

void ofxSvm::train()
{
    svm_problem prob;
    
    prob.l = mData.size();
    prob.y = new double[prob.l];
    {
        data_type::iterator it = mData.begin();
        int i = 0;
        while (it != mData.end())
        {
            prob.y[i] = it->first;
            ++it; ++i;
        }
    }
    
    if(mParam.gamma == 0)
    {
        mParam.gamma = 1.0 / mDimension;
    }
    
    int nodeLength = mDimension + 1;
    svm_node* node = new svm_node[prob.l * nodeLength];
    prob.x = new svm_node*[prob.l];
    {
        data_type::iterator it = mData.begin();
        int i = 0;
        while (it != mData.end())
        {
            prob.x[i] = node + i * nodeLength;
            for (int j = 0; j < mDimension; ++j)
            {
                prob.x[i][j].index = j + 1;
                prob.x[i][j].value = it->second[j];
            }
            prob.x[i][mDimension].index = -1; // delimiter
            ++it; ++i;
        }
    }
    
    ofLogVerbose(LOG_MODULE, "Start train...");
    
    mModel = svm_train(&prob, &mParam);
    
    ofLogVerbose(LOG_MODULE, "Finished train!");
    
    delete[] node;
    delete[] prob.x;
    delete[] prob.y;
}

int ofxSvm::predict(vector<double>& testVec)
{
    if (mModel == NULL)
    {
        ofLogError(LOG_MODULE, "null model, befor do train or load model file");
        return 0;
    }
    if (testVec.size() != mDimension)
    {
        ofLogError(LOG_MODULE, "diffetent dimension");
        return 0;
    }
    
    svm_node* node = new svm_node[mDimension + 1];
    for (int i = 0; i < mDimension; ++i)
    {
        node[i].index = i + 1;
        node[i].value = testVec[i];
    }
    node[mDimension].index = -1;
    
    int res = static_cast<int>(svm_predict(mModel, node));
    
    delete[] node;
    return res;
}

void ofxSvm::saveModel(const string &filename)
{
    if (mModel == NULL)
    {
        ofLogError(LOG_MODULE, "null model, befor do train or load model file");
        return;
    }
    svm_save_model(ofToDataPath(filename).c_str(), mModel);
}

void ofxSvm::loadModel(const string &filename)
{
    mModel = svm_load_model(ofToDataPath(filename).c_str());
}

vector<int> ofxSvm::getSupportVectorIndex()
{
    vector<int> dst;
    
    if (mModel == NULL)
    {
        ofLogError(LOG_MODULE, "null model, befor do train or load model file");
        return dst;
    }
    
    const int num = svm_get_nr_sv(mModel);
    int* indices = new int[num];
    svm_get_sv_indices(mModel, indices);
    for(int i = 0; i < num; ++i )
    {
        dst.push_back(indices[i] - 1);
    }
    delete[] indices;
    return dst;
}
