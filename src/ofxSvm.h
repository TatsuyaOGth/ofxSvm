#pragma once

#include "ofMain.h"
#include "svm.h"

class ofxSvm
{
    typedef multimap<int, vector<double> > data_type;
    
    svm_parameter   mParam;
    svm_model       *mModel;
    
    int             mDimension;
    data_type       mData;
    
    
    void checkDimension(int length);
    
public:
    ofxSvm();
    virtual ~ofxSvm();
    
    int     addData(int label, vector<double>& vec);
    int     addData(int label, double* vec, int length);
    void    creatData();
    
    void    train();
    int     predict(vector<double>& testVec);
    
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
