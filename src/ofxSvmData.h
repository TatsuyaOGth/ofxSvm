#ifndef ofxSvmExample_ofxSvmData_h
#define ofxSvmExample_ofxSvmData_h

#include "ofMain.h"

/**
 *  SVM dataset class
 *
 *  - add dataset in lavel and features<br/>
 *  - scaling and make scaled dataset<br/>
 *  - get dataset or get scaled dataset<br/>
 *  - export dataset/scaled dataset to file
 */
class ofxSvmData
{
public:
    ofxSvmData();
    ~ofxSvmData();
    
    typedef multimap<int, vector<double> > dataset_type;
    
    dataset_type& getDataRef(){ return mData; }
    
    void setData(const int label, const int feature_num, ...);
        
    void setData(const int label, vector<double>& features_vector)
    {
        mData.insert(make_pair(label, features_vector));
    }
    
    template<typename T>
    void setData(const int label, vector<T>& features_vector)
    {
        vector<double> tmp;
        for (const auto& e : features_vector)
            tmp.push_back(static_cast<double>(e));
        mData.insert(make_pair(label, tmp));
    }
    
    void exportDataset(const string& filename);
    
    static bool scaling(const char* data_file_name, const char* output_file_name,
                        const double x_lower, const double x_upper,
                        const double y_lower, const double y_upper);
    
private:
    dataset_type mData;

};

#endif
