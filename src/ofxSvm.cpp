#include "ofxSvm.h"
#include "svm-scale.h"
#include "svm-train.h"
#include "svm-predict.h"


bool ofxSvm::train(const string& dataset_file_name, const string& output_model_file_name)
{
    if (mGamma == 0)
    {
        mGamma = 1.0 / mDataset.begin()->second.size();
    }
    
    set_svm_train_parameters((int)mSvmType, (int)mKernelType, mDegree, mGamma, mCoef0, mNu, mCachesize,
                             mCost, mTolerance, mEpsilon, mUseShrinking ? 1 : 0, mProbabilityEstimates ? 1 : 0, mCrossValidationNr);
    
    ofFile inf(dataset_file_name);
    ofFile outf(output_model_file_name);
    int res = execute_svm_train(inf.getAbsolutePath().c_str(), outf.getAbsolutePath().c_str());
    return res == 0;
}

bool ofxSvm::scaling(const string& data_file_name, const string& output_file_name,
                     const double x_lower, const double x_upper,
                     const double y_lower, const double y_upper)
{
    ofFile inf(data_file_name);
    ofFile outf(output_file_name);
    int res = svm_scale(inf.getAbsolutePath().c_str(), outf.getAbsolutePath().c_str(), x_lower, x_upper, y_lower, y_upper, NULL, NULL);
    return res == 0;
}

bool ofxSvm::predict(const string& test_file_name, const string& model_file_name,
                     const string& result_file_name, const bool predict_probability)
{
    ofFile f1(test_file_name);
    ofFile f2(model_file_name);
    ofFile f3(result_file_name);
    int res = execute_svm_predict(f1.getAbsolutePath().c_str(), f2.getAbsolutePath().c_str(), f3.getAbsolutePath().c_str(), predict_probability ? 1 : 0);
    return res == 0;
}



void ofxSvm::setData(const int lavel, vector<double> &features)
{
    bool ret = true;
    if (!mDataset.empty())
    {
        int inSize = features.size();
        for (dataset_type::iterator it = mDataset.begin(); it != mDataset.end(); it++) {
            ret = inSize == it->second.size();
        }
    }
    if (!ret) ofLogWarning("ofxSvm::setData") << "in different size";
    mDataset.insert(make_pair(lavel, features));
}

void ofxSvm::dumpDataset()
{
    for (const auto& e : mDataset)
    {
        cout << e.first << " ";
        for (int i = 0; i < e.second.size(); ++i)
        {
            cout << (i+1) << ":" << e.second[i] << " ";
        }
        cout << "\n";
    }
}

void ofxSvm::dumpDataset(const string& data_file_name)
{
    ofBuffer buffer = ofBufferFromFile(data_file_name);
    while(buffer.isLastLine() == false)
    {
        cout << buffer.getNextLine();
        cout << "\n";
    }
}

void ofxSvm::exportDataset(const string& filename)
{
    ofFile file(filename, ofFile::WriteOnly);
    for (const auto& e : mDataset)
    {
        file << e.first << " ";
        for (int i = 0; i < e.second.size(); ++i)
        {
            file << (i+1) << ":" << e.second[i] << " ";
        }
        file << "\n";
    }
    file.close();
}



