#ifndef ofxSvmExample_svm_predict_h
#define ofxSvmExample_svm_predict_h

#ifdef __cplusplus
extern "C" {
#endif
    
    extern int execute_svm_predict(const char* test_file, const char* model_file, const char* output_file, const int _predict_probability);
    
#ifdef __cplusplus
}
#endif

#endif
