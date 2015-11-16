#pragma once

#include "ofMain.h"
#include "svm.h"

class ofxSvm
{
public:
    typedef vector<double> vector_type;
    typedef multimap<double, vector_type> data_type;
    
protected:
    svm_parameter   mParam;
    svm_model       *mModel;
    
    int             mDimension;
    data_type       mData;
    vector_type     mTempVector;
    
    void            checkDimension(int length);
    static void     printStdOut(const char *s);

public:
    struct ScaleParameter
    {
        bool isEnable;
        double x_lower, x_upper, y_lower, y_upper;
        double y_min, y_max;
        vector_type feature_max, feature_min;
    };
    struct ScaleParameter mScaleParameter;
    
public:
    ofxSvm();
    virtual ~ofxSvm();
    
    int     addData(double label, vector<double>& vec);
    int     addData(double label, double* vec, int length);
    
    template<typename T>
    int addData(double label, vector<T>& vec)
    {
        mTempVector.clear();
        for (const auto& e : vec) mTempVector.push_back(static_cast<double>(e));
        return addData(label, mTempVector);
    }
    
    void    clearData();
    
    bool    scale(double lower, double upper, double y_lower = 0, double y_upper = 0);
    bool    hasScaleParameter() { return mScaleParameter.isEnable; }
    
    void    train();
    int     predict(const vector<double>& testVec);
    
    void    saveModel(const string& filename);
    void    loadModel(const string& filename);
    
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
};
