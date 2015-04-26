#ifndef ofxSvmExample_svm_train_h
#define ofxSvmExample_svm_train_h

#ifdef __cplusplus
extern "C" {
#endif
    
    extern int execute_svm_train(const char* _data_file_name, const char* _model_file_name);
    
    extern void set_svm_train_parameters(const int _svm_type,
                                         const int _kernel_type,
                                         const int _degree,
                                         const double _gamma,
                                         const double _coef0,
                                         const double _nu,
                                         const double _cache_size,
                                         const double _C,
                                         const double _eps,
                                         const double _p,
                                         const int _shrinking,
                                         const int _probability,
                                         const int _cross_validation_nr);
    
    extern void set_svm_train_weight(const int lavel, const double weight);
    
    

#ifdef __cplusplus
}
#endif
#endif
