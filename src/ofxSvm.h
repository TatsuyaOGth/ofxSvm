#pragma once

#include "ofMain.h"
#include "svm.h"

class ofxSvm
{
public:
    class Data
    {
        friend class ofxSvm;
        
        multimap<double, vector<double> > mData;
        int mDimension;
        
        struct ScaleParameter
        {
            bool isEnable;
            double x_lower, x_upper, y_lower, y_upper;
            double y_min, y_max;
            vector<double> feature_max, feature_min;
        };
        struct ScaleParameter mScaleParameter;
        
        void checkDimension(int length);
        
    public:
        Data();
        
        int     add(double label, vector<double>& vec);
        int     add(double label, double* vec, int length);
        
        void    clear();
        
        bool    scale(double lower, double upper, double y_lower = 0, double y_upper = 0);
        bool    hasScaleParameter() { return mScaleParameter.isEnable; }

    };
    
    // including namespace
    enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
    enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

    
public:
    ofxSvm();
    virtual ~ofxSvm();
    
    void    train(const Data& data);
    vector<double>  predict(const Data& data);
    
    void    saveModel(const string& filename);
    void    loadModel(const string& filename);
    
    void    clear();
    
    vector<int> getSupportVectorIndex();
    
    inline void setSvmType(const int type)      { mParam.svm_type = type;           }
    inline void setKernelType(const int type)   { mParam.kernel_type = type;        }
    inline void setDegree(const double v)       { mParam.degree = v;                }
    inline void setGamma(const double v)        { mParam.gamma = v;                 }
    inline void setCoef0(const double v)        { mParam.coef0 = v;                 }
    inline void setCost(const double v)         { mParam.C = v;                     }
    inline void setNu(const double v)           { mParam.nu = v;                    }
    inline void setCachssize(const double v)    { mParam.cache_size = v;            }
    inline void setEpsilon(const double v)      { mParam.eps = v;                   }
    inline void setP(const double v)            { mParam.p = v;                     }
    inline void setNrWeight(const int v)        { mParam.nr_weight = v;             }
    inline void setShrinking(const bool v)      { mParam.shrinking = v ? 1 : 0;     }
    inline void setProbability(const bool v)    { mParam.probability = v ? 1 : 0;   }
    inline void setWeightLabel(int* v)          { mParam.weight_label = v;          }
    inline void setWeight(double* v)            { mParam.weight = v;                }
    void        defaultParams();
    
protected:
    svm_parameter   mParam;
    svm_model       *mModel;
    Data const      *mTrainData;
    
    static void     printStdOut(const char *s);
};
