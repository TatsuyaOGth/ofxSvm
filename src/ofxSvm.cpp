#include "ofxSvm.h"
#include "svm-utils.h"
#include "svm-scale.h"

bool ofxSvm::train(const string& dataset_file_name, const string& output_model_file_name)
{
    if (nr_fold != 0)
    {
        cross_validation = 1;
        if(nr_fold < 2)
        {
            ofLogError("ofxSvm::train") << "n-fold cross validation: n must >= 2" << endl;
            return false;
        }
    }
    
    ofFile fin(dataset_file_name);
    struct svm_problem prob;
    
    int res = read_problem(fin.getAbsolutePath().c_str(), &prob, &param);
    if (res != 0)
        goto ERROR_RETURN;
    
    model = svm_train(&prob, &param);
    
    if (!output_model_file_name.empty())
    {
        if (!saveModelData(output_model_file_name))
        {
            goto ERROR_RETURN;
        }
    }
    
    free(prob.y);
    free(prob.x);
    return true;
    
ERROR_RETURN:
    free(prob.y);
    free(prob.x);
    return false;
}

bool ofxSvm::predict(const string& test_file_name, const string& model_file_name,
                     const string& result_file_name, const bool predict_probability)
{
    ofFile f1(test_file_name);
    ofFile f2(model_file_name);
    ofFile f3(result_file_name);
//    int res = execute_svm_predict(f1.getAbsolutePath().c_str(), f2.getAbsolutePath().c_str(), f3.getAbsolutePath().c_str(), predict_probability ? 1 : 0);
//    return res == 0;
    return true;
}

bool ofxSvm::saveModelData(const string &save_file_name)
{
    ofFile fout(save_file_name);
    int res = svm_save_model(fout.getAbsolutePath().c_str(), model);
    return res != 0;
}

bool ofxSvm::loadModelData(const string &model_file_name)
{
    ofFile fin(model_file_name);
    model = svm_load_model(fin.getAbsolutePath().c_str());
    return model != NULL;
}



bool ofxSvm::scaling(const char *data_file_name, const char *output_file_name, const double x_lower, const double x_upper, const double y_lower, const double y_upper)
{
    ofFile inf(data_file_name);
    ofFile outf(output_file_name);
    int res = svm_scale(inf.getAbsolutePath().c_str(), outf.getAbsolutePath().c_str(), x_lower, x_upper, y_lower, y_upper);
    return res == 0;
}

