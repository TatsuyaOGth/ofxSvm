#pragma once

#include "ofMain.h"
#include "svm.h"

class ofxSvm
{
private:
    struct svm_parameter param; ///< Training parameters
    struct svm_model* model; ///< Model data
    int cross_validation;
    int nr_fold;
    
public:
    ofxSvm()
    {
        param.svm_type = C_SVC;
        param.kernel_type = RBF;
        param.degree = 3;
        param.gamma = 0; // 1/n
        param.coef0 = 0;
        param.C = 1;
        param.nu = 0.5;
        param.cache_size = 100;
        param.eps = 0.1;
        param.p = 0.001;
        param.shrinking = 1;
        param.probability = 0;
        cross_validation = 0;
    }
    
    virtual ~ofxSvm()
    {
        if (model) svm_free_and_destroy_model(&model);
    }
    
    
        
    bool train(const string& dataset_file_name, const string& output_model_file_name = "");
    
    bool predict(const string& test_file_name, const string& model_file_name,
                 const string& result_file_name, const bool predict_probability = false);
    
    
    void setSvmType(const int type) { param.svm_type = type; }
    void setKernelType(const int type) { param.kernel_type = type; }
    void setDegree(const float v) { param.degree = v; }
    void setGamma(const float v) { param.gamma = v; }
    void setCoef0(const float v) { param.coef0 = v; }
    void setCost(const float v) { param.C = v; }
    void setNu(const float v) { param.nu = v; }
    void setCachssize(const float v) { param.cache_size = v; }
    void setEpsilon(const float v) { param.eps = v; }
    void setUseShrinking(const bool v) { param.shrinking = v ? 1 : 0; }
    void setNumClossValidation(const int v) { nr_fold = v; }
    
    bool saveModelData(const string& save_file_name);
    bool loadModelData(const string& model_file_name);
    
    
public:
    
    // utility method

    template<class Iterator>
    static void writeFeatures(Iterator first, Iterator last, ofFile& file)
    {
        int i = 1;
        first++;
        while (first != last)
        {
            file << i << ":" << *first << " ";
            i++;
            first++;
        }
        file << "\n";
    }
    
    template<typename container>
    static void saveToFileDataset(const string& save_file_name, const container& dataset)
    {
        ofFile file(save_file_name, ofFile::WriteOnly);
        for (const auto& data : dataset)
        {
            file << static_cast<int>(data[0]) << " ";
            writeFeatures(data.begin(), data.end(), file);
        }
        file.close();
    }

    
    static bool scaling(const char *data_file_name, const char *output_file_name,
                        const double x_lower, const double x_upper,
                        const double y_lower, const double y_upper);

};
