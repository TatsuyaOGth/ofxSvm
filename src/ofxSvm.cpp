#include "ofxSvm.h"

#define LOG_MODULE "ofxSvm::"+string(__FUNCTION__)
#define max(x,y) (((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))


ofxSvm::Data::Data() :  mDimension(0)
{
    mScaleParameter.isEnable = false;
}

void ofxSvm::Data::checkDimension(int length)
{
    if (mDimension > 0 && mDimension != length)
    {
        ofLogWarning(LOG_MODULE, "got different dimensions, set data");
        mData.clear();
    }
}

int ofxSvm::Data::add(double label, vector<double>& vec)
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
    
    mScaleParameter.isEnable = false;
    return mData.size();
}

int ofxSvm::Data::add(double label, double *vec, int length)
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
    
    mScaleParameter.isEnable = false;
    return mData.size();
}

void ofxSvm::Data::clear()
{
    mData.clear();
    mDimension = 0;
    mScaleParameter.isEnable = false;
}

bool ofxSvm::Data::scale(double lower, double upper, double y_lower, double y_upper)
{
    bool y_scaling = false;
    if (y_lower != 0 && y_upper != 0) y_scaling = true;
    
    if (!(upper > lower) || (y_scaling && !(y_upper > y_lower)))
    {
        ofLogError(LOG_MODULE) << "inconsistent lower/upper specification";
        return false;
    }
    
    if (mData.empty())
    {
        ofLogError(LOG_MODULE) << "vector object has not data";
        return false;
    }
    
    
    /* assumption: min index of attributes is 1 */
    /* pass 1: find out max index of attributes */
    int max_index = 0;
    int min_index = 1;
    long int num_nonzeros = 0;
    long int new_num_nonzeros = 0;
    
    for (const auto& data : mData)
    {
        max_index = max(max_index, data.second.size());
        min_index = min(min_index, data.second.size());
        num_nonzeros++;
    }
    
    if(min_index < 1)
    {
        ofLogWarning(LOG_MODULE) << "minimal feature index is " << min_index << ", but indices should start from 1";
    }
    
    
    /* pass 2: find out min/max value */
    vector<double> feature_max(max_index + 1);
    vector<double> feature_min(max_index + 1);
    double y_max = -DBL_MAX;
    double y_min =  DBL_MAX;
    
    for (int i = 0; i <= max_index; i++)
    {
        feature_max[i] = -DBL_MAX;
        feature_min[i] =  DBL_MAX;
    }
    
    for (const auto& data : mData)
    {
        auto label = data.first;
        
        y_max = max(y_max, label);
        y_min = min(y_min, label);
        
        int index = 1;
        for (int i = 0; i < data.second.size(); ++i)
        {
            feature_max[i] = max(feature_max[i], data.second[i]);
            feature_min[i] = min(feature_min[i], data.second[i]);
        }
        
        for(int i = data.second.size(); i <= max_index; i++)
        {
            feature_max[i] = max(feature_max[i], 0);
            feature_min[i] = min(feature_min[i], 0);
        }
    }
    
    /* pass 3: scale */
    multimap<double, vector<double> > scaled_data;
    for (const auto& data : mData)
    {
        int next_index = 1;
        auto scaled_label = data.first;
        
        if (y_scaling)
        {
            auto target = data.first;
            
            if (target == y_min)
                target = y_lower;
            else if(target == y_max)
                target = y_upper;
            else target = y_lower + (y_upper-y_lower) *
                (target - y_min)/(y_max-y_min);
            scaled_label = target;
        }
        
        vector<double> scaled_vector;
        for (int i = 0; i < data.second.size(); ++i)
        {
            /* skip single-valued attribute */
            if (feature_max[i] == feature_min[i]) continue;
            
            auto value = data.second[i];
            double dstValue = 0;
            if (value == feature_min[i])
                dstValue = lower;
            else if(value == feature_max[i])
                dstValue = upper;
            else
                dstValue = lower + (upper-lower) * (value-feature_min[i]) / (feature_max[i]-feature_min[i]);
            
            if (dstValue != 0)
            {
                scaled_vector.push_back(dstValue);
                new_num_nonzeros++;
            }
        }
        
        for (int i = data.second.size(); i <= max_index; i++)
        {
            auto value = 0;
            double dstValue = 0;
            if (value == feature_min[i])
                dstValue = lower;
            else if(value == feature_max[i])
                dstValue = upper;
            else
                dstValue = lower + (upper-lower) * (value-feature_min[i]) / (feature_max[i]-feature_min[i]);
            
            if (scaled_vector.back() != 0)
            {
                scaled_vector.push_back(dstValue);
                new_num_nonzeros++;
            }
        }
        
        scaled_data.insert(make_pair(scaled_label, scaled_vector));
    }
    
    mData.swap(scaled_data);
    
    if (new_num_nonzeros > num_nonzeros)
    {
        ofLogWarning(LOG_MODULE)
        << "original #nonzeros " << num_nonzeros << ", new #nonzeros " << new_num_nonzeros
        << ", Use first augument is 0 if many original feature values are zeros";
    }
    
    // save scale parameters
    mScaleParameter.y_lower = y_lower;
    mScaleParameter.y_upper = y_upper;
    mScaleParameter.x_lower = lower;
    mScaleParameter.x_upper = upper;
    mScaleParameter.y_min = y_min;
    mScaleParameter.y_max = y_max;
    mScaleParameter.feature_min = feature_min;
    mScaleParameter.feature_max = feature_max;
    mScaleParameter.isEnable = true;
    
    return true;
}




ofxSvm::ofxSvm() : mModel(NULL), mTrainData(NULL)
{
    defaultParams();
    svm_set_print_string_function(ofxSvm::printStdOut);
}

ofxSvm::~ofxSvm()
{
    clear();
}

void ofxSvm::clear()
{
    svm_destroy_param(&mParam);
    svm_free_and_destroy_model(&mModel);
}

void ofxSvm::defaultParams()
{
    mParam.svm_type     = C_SVC;
    mParam.kernel_type  = RBF;
    mParam.degree       = 3;
    mParam.gamma        = -DBL_MAX; // 1/n
    mParam.coef0        = 0;
    mParam.C            = 1;
    mParam.nu           = 0.5;
    mParam.cache_size   = 100;
    mParam.eps          = 0.001;
    mParam.p            = 0.1;
    mParam.shrinking    = 1;
    mParam.probability  = 0;
    mParam.weight_label = NULL;
    mParam.weight       = NULL;
}


void ofxSvm::printStdOut(const char *s)
{
    fputs(s, stdout);
    fflush(stdout);
}

void ofxSvm::train(const Data& data)
{
    svm_problem prob;
    
    const multimap<double, vector<double> >& v = data.mData;
    
    prob.l = v.size();
    prob.y = new double[prob.l];
    {
        multimap<double, vector<double> >::const_iterator it = v.begin();
        int i = 0;
        while (it != v.end())
        {
            prob.y[i] = it->first;
            ++it; ++i;
        }
    }
    
    if (mParam.gamma == -DBL_MAX)
    {
        mParam.gamma = 1.0 / data.mDimension;
        ofLogVerbose(LOG_MODULE, "Gamma set 1/n: " + ofToString(mParam.gamma));
    }
    
    int nodeLength = data.mDimension + 1;
    svm_node* node = new svm_node[prob.l * nodeLength];
    prob.x = new svm_node*[prob.l];
    {
        multimap<double, vector<double> >::const_iterator it = v.begin();
        int i = 0;
        while (it != v.end())
        {
            if (it->second.empty() == false)
            {
                prob.x[i] = node + i * nodeLength;
                for (int j = 0; j < data.mDimension; ++j)
                {
                    prob.x[i][j].index = j + 1;
                    prob.x[i][j].value = it->second[j];
                }
            }
            prob.x[i][data.mDimension].index = -1; // delimiter
            ++it; ++i;
        }
    }
    
    ofLogVerbose(LOG_MODULE, "Checking parameters");
    const char* check_res = svm_check_parameter(&prob, &mParam);
    if (check_res != NULL)
    {
        ofLogError(LOG_MODULE, ofToString(check_res));
        delete[] node;
        delete[] prob.x;
        delete[] prob.y;
        return;
    }
    
    ofLogVerbose(LOG_MODULE, "Start train...");
    
    mModel = svm_train(&prob, &mParam);
    
    ofLogVerbose(LOG_MODULE, "Finish");
    
    delete[] node;
    delete[] prob.x;
    delete[] prob.y;
    
    mTrainData = &data;
}

vector<double> ofxSvm::predict(const Data& data)
{
    vector<double> res;
    
    if (mTrainData == NULL)
    {
        ofLogError(LOG_MODULE, "did not trained yet");
        return res;
    }
    if (mModel == NULL)
    {
        ofLogError(LOG_MODULE, "null model, befor do train or load model file");
        return res;
    }
    if (svm_check_probability_model(mModel))
    {
        ofLogError(LOG_MODULE, "provavility model is not available");
        return res;
    }
    if (data.mScaleParameter.isEnable != mTrainData->mScaleParameter.isEnable)
    {
        ofLogWarning(LOG_MODULE, "different scale");
    }
    
    multimap<double, vector<double> >::const_iterator it = data.mData.begin();
    while (it != data.mData.end())
    {
        const vector<double>& testVec = it->second;
        
        if (testVec.size() != mTrainData->mDimension)
        {
            ofLogError(LOG_MODULE, "diffetent dimension");
            res.push_back(-DBL_MAX);
            ++it;
            continue;
        }
        
        vector<double> testVector(testVec);
        
        svm_node* node = new svm_node[data.mDimension + 1];
        for (int i = 0; i < data.mDimension; ++i)
        {
            node[i].index = i + 1;
            node[i].value = testVector[i];
        }
        node[data.mDimension].index = -1;
        
        res.push_back( svm_predict(mModel, node) );
        delete[] node;
        
        ++it;
    }
    
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
    vector<int> indices(num);
    svm_get_sv_indices(mModel, indices.data());
    for(int i = 0; i < num; ++i )
    {
        dst.push_back(indices[i] - 1);
    }
    return dst;
}
