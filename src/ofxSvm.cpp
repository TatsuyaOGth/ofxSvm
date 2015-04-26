#include "ofxSvm.h"
#include "svm-utils.h"



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

int ofxSvm::classify(ofxSvmData &test_svm_data)
{
    ofxSvmData::dataset_type& data = test_svm_data.getDataRef();
    int i = 0;
    struct svm_node *x;
    int max_nr_attr = 64;
    
    x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
    
    for (auto& v : data)
    {
        if(i>=max_nr_attr-1)	// need one more for index = -1
        {
            max_nr_attr *= 2;
            x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
        }
        
        for (auto& feature : v.second)
        {
            x[i].index = i + 1;
            x[i].value = feature;
            
            ++i;
        }
        
//        idx = strtok(NULL,":");
//        val = strtok(NULL," \t");
//        
//        if(val == NULL)
//            break;
//        errno = 0;
//        x[i].index = (int) strtol(idx,&endptr,10);
//        if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
//            return exit_input_error(total+1);
//        else
//            inst_max_index = x[i].index;
//        
//        errno = 0;
//        x[i].value = strtod(val,&endptr);
//        if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
//            return exit_input_error(total+1);
        
        
    }
}
