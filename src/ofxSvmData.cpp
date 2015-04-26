#include "ofxSvmData.h"

ofxSvmData::ofxSvmData()
{
}

ofxSvmData::~ofxSvmData()
{
}

void ofxSvmData::setData(const int label, const int feature_num, ...)
{
    if (feature_num < 1)
    {
        return;
    }
    vector<double> tmp;
    va_list args;
    va_start(args , feature_num);
    for (int i = 0 ; i < feature_num ; i++)
    {
        tmp.push_back(va_arg(args , double));
    }
    va_end(args);
    mData.insert(make_pair(label, tmp));
}

void ofxSvmData::exportDataset(const string &filename)
{
    ofFile file(filename, ofFile::WriteOnly);
    for (const auto& e : mData)
    {
        file << e.first << " ";
        for (int i = 0; i < e.second.size(); ++i)
        {
            file << (i+1) << ":" << e.second[i] << " ";
        }
        file << "\n";
    }
    file.close();
}

bool ofxSvmData::scaling(const char *data_file_name, const char *output_file_name,
                         const double x_lower, const double x_upper,
                         const double y_lower, const double y_upper)
{
    ofFile inf(data_file_name);
    ofFile outf(output_file_name);
//    int res = svm_scale(inf.getAbsolutePath().c_str(), outf.getAbsolutePath().c_str(), x_lower, x_upper, y_lower, y_upper);
//    return res == 0;
    return true;
}
