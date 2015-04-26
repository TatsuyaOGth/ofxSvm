#ifndef ofxSvmExample_svm_scale_h
#define ofxSvmExample_svm_scale_h

#ifdef __cplusplus
extern "C" {
#endif
    
    extern int svm_scale(const char* data_file, const char* dst_file,
                         const double _lower, const double _upper,
                         const double _y_lower, const double _y_upper);
#ifdef __cplusplus
}
#endif

#endif
