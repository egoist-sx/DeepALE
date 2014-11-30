//
//  kyheader.h
//  ImageTracking
//
//  Created by bittnt on 29/11/2014.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#ifndef ImageTracking_kyheader_h
#define ImageTracking_kyheader_h
#include <stdio.h>
#include <assert.h>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <math.h>
#include <time.h>
#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#ifdef WIN32
/* windows stuff */
#else
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned int UNINT32;
typedef bool BOOL;
typedef void *HANDLE;
typedef unsigned char byte;
#endif
typedef std::vector<int> vecI;
typedef const std::string CStr;
typedef const cv::Mat CMat;
typedef std::vector<std::string> vecS;
typedef std::vector<cv::Mat> vecM;
typedef std::vector<float> vecF;
typedef std::vector<double> vecD;

enum{CV_FLIP_BOTH = -1, CV_FLIP_VERTICAL = 0, CV_FLIP_HORIZONTAL = 1};
#define _S(str) ((str).c_str())
#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)
#define CV_Assert_(expr, args) \
{\
    if(!(expr)) {\
        string msg = cv::format args; \
        printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__); \
        cv::error(cv::Exception(CV_StsAssert, msg, __FUNCTION__, __FILE__, __LINE__) ); }\
}

// Return -1 if not in the list
template<typename T>
static inline int findFromList(const T &word, const std::vector<T> &strList) {size_t idx = find(strList.begin(), strList.end(), word) - strList.begin(); return idx < strList.size() ? idx : -1;}
template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...
template<class T, int D> inline T vecSqrDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqr(v1[i] - v2[i]); return s;} // out of range risk for T = byte, ...
template<class T, int D> inline T    vecDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...

inline cv::Rect Vec4i2Rect(cv::Vec4i &v){return cv::Rect(cv::Point(v[0] - 1, v[1] - 1), cv::Point(v[2], v[3])); }
#ifdef __WIN32
#define INT64 long long
#else
#define INT64 long
typedef unsigned long UINT64;
#endif

#define __POPCNT__
#include <immintrin.h>
#include <popcntintrin.h>
#ifdef __WIN32
# include <intrin.h>
# define POPCNT(x) __popcnt(x)
# define POPCNT64(x) __popcnt64(x)
#endif
#ifndef __WIN32
# define POPCNT(x) __builtin_popcount(x)
# define POPCNT64(x) __builtin_popcountll(x)
#endif


#endif
