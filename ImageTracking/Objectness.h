//
//  Objectness.h
//  ImageTracking
//
//  Created by bittnt on 29/11/2014.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#ifndef __ImageTracking__Objectness__
#define __ImageTracking__Objectness__

#include <stdio.h>
#include "kyheader.h"
#include "ValStructVec.h"
#include "FilterTIG.h"
#include "DataSetVOC.h"
class Objectness{
public:
    // base for window size quantization, feature window size (W, W), and non-maximal suppress size NSS
    Objectness(DataSetVOC &voc, double base = 2, int W = 8, int NSS = 2);
    Objectness(DataSetVOC &voc, std::string &model_name, double base = 2, int W = 8, int NSS = 2);

    ~Objectness(void);
    
    // Load trained model.
    int loadTrainedModel(std::string modelName = ""); // Return -1, 0, or 1 if partial, none, or all loaded
    
    // Get potential bounding boxes, each of which is represented by a Vec4i for (minX, minY, maxX, maxY).
    // The trained model should be prepared before calling this function: loadTrainedModel() or trainStageI() + trainStageII().
    // Use numDet to control the final number of proposed bounding boxes, and number of per size (scale and aspect ratio)
    void getObjBndBoxes(const cv::Mat &img3u, ValStructVec<float, cv::Vec4i> &valBoxes, int numDetPerSize = 120);
    

    void setColorSpace(int clr, std::string &model_name);
    void setColorSpace(int clr = MAXBGR);
    
    // Read matrix from binary file
    static bool matRead( const std::string& filename, cv::Mat& M);
    
    enum {MAXBGR, HSV, G};
    
    
    static void meanStdDev(CMat &data1f, cv::Mat &mean1f, cv::Mat &stdDev1f);
    
    inline static float LoG(float x, float y, float delta) {float d = -(x*x+y*y)/(2*delta*delta);  return -1.0f/((float)(CV_PI)*pow(delta, 4)) * (1+d)*exp(d);} // Laplacian of Gaussian
    static cv::Mat aFilter(float delta, int sz);
    
    
private: // Parameters
    const double _base, _logBase; // base for window size quantization
    const int _W; // As described in the paper: #Size, Size(_W, _H) of feature window.
    const int _NSS; // Size for non-maximal suppress
    const int _maxT, _minT, _numT; // The minimal and maximal dimensions of the template
    
    int _Clr; //
    static const char* _clrName[3];
    
    DataSetVOC &_voc; // The dataset for training, testing
    std::string _modelName, _trainDirSI, _bbResDir;
    
    vecI _svmSzIdxs; // Indexes of active size. It's equal to _svmFilters.size() and _svmReW1f.rows
    cv::Mat _svmFilter; // Filters learned at stage I, each is a _H by _W CV_32F matrix
    FilterTIG _tigF; // TIG filter
    cv::Mat _svmReW1f; // Re-weight parameters learned at stage II.
    
    
    
private: // Help functions
    
    bool filtersLoaded() {int n = _svmSzIdxs.size(); return n > 0 && _svmReW1f.size() == cv::Size(2, n) && _svmFilter.size() == cv::Size(_W, _W);}
    
    int gtBndBoxSampling(const cv::Vec4i &bbgt, std::vector<cv::Vec4i> &samples, vecI &bbR);
    
    cv::Mat getFeature(CMat &img3u, const cv::Vec4i &bb); // Return region feature
    
    inline double maxIntUnion(const cv::Vec4i &bb, const std::vector<cv::Vec4i> &bbgts) {double maxV = 0; for(size_t i = 0; i < bbgts.size(); i++) maxV = std::max(maxV, DataSetVOC::interUnio(bb, bbgts[i])); return maxV; }
    
    // Convert VOC bounding box type to OpenCV Rect
    inline cv::Rect pnt2Rect(const cv::Vec4i &bb){int x = bb[0] - 1, y = bb[1] - 1; return cv::Rect(x, y, bb[2] -  x, bb[3] - y);}
    
    // Template length at quantized scale t
    inline int tLen(int t){return cvRound(pow(_base, t));}
    
    // Sub to quantization index
    inline int sz2idx(int w, int h) {w -= _minT; h -= _minT; CV_Assert(w >= 0 && h >= 0 && w < _numT && h < _numT); return h * _numT + w + 1; }
    inline std::string strVec4i(const cv::Vec4i &v) const {return cv::format("%d, %d, %d, %d", v[0], v[1], v[2], v[3]);}
    
    void generateTrianData();
    void trainStageI();
    void trainStateII(int numPerSz = 100);
    void predictBBoxSI(CMat &mag3u, ValStructVec<float, cv::Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ = 100, bool fast = true);
    void predictBBoxSII(ValStructVec<float, cv::Vec4i> &valBoxes, const vecI &sz);
    
    // Calculate the image gradient: center option as in VLFeat
    void gradientMag(CMat &imgBGR3u, cv::Mat &mag1u);
    
    static void gradientRGB(const cv::Mat &bgr3u, cv::Mat &mag1u);
    static void gradientGray(const cv::Mat &bgr3u, cv::Mat &mag1u);
    static void gradientHSV(const cv::Mat &bgr3u, cv::Mat &mag1u);
    static void gradientXY(const cv::Mat &x1i, const cv::Mat &y1i, cv::Mat &mag1u);
    
    static inline int bgrMaxDist(const cv::Vec3b &u, const cv::Vec3b &v) {
        int b = std::min(std::abs(u[0]-v[0]),255), g = std::min(std::abs(u[1]-v[1]),255), r = std::min(std::abs(u[2]-v[2]),255);
        b = std::max(b,g);
        return std::max(b,r);
    }
    static inline int vecDist3b(const cv::Vec3b &u, const cv::Vec3b &v) {return std::abs(u[0]-v[0]) + std::abs(u[1]-v[1]) + std::abs(u[2]-v[2]);}
    
    //Non-maximal suppress
    static void nonMaxSup(const cv::Mat &matchCost1f, ValStructVec<float, cv::Point> &matchCost, int NSS = 1, int maxPoint = 50, bool fast = true);
    
    static void PrintVector(FILE *f, const vecD &v, CStr &name);
    
    vecD getVector(CMat &t1f);
};
#endif /* defined(__ImageTracking__Objectness__) */
