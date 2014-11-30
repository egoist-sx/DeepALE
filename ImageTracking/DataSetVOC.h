//
//  DataSetVOC.h
//  ImageTracking
//
//  Created by bittnt on 29/11/2014.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#ifndef __ImageTracking__DataSetVOC__
#define __ImageTracking__DataSetVOC__
#include "kyheader.h"
#include <stdio.h>


struct DataSetVOC
{
    DataSetVOC(CStr &wkDir);
    ~DataSetVOC(void);
    
    // Organization structure data for the dataset
    std::string wkDir; // Root working directory, all other directories are relative to this one
    std::string resDir, localDir; // Directory for saving results and local data
    std::string imgPathW, annoPathW; // Image and annotation path
    
    // Information for training and testing
    int trainNum, testNum;
    vecS trainSet, testSet; // File names (NE) for training and testing images
    vecS classNames; // Object class names
    std::vector<std::vector<cv::Vec4i>> gtTrainBoxes, gtTestBoxes; // Ground truth bounding boxes for training and testing images
    std::vector<vecI> gtTrainClsIdx, gtTestClsIdx; // Object class indexes
    
    
    // Load annotations
    void loadAnnotations();
        
    static inline double interUnio(const cv::Vec4i &box1, const cv::Vec4i &box2);
    
    // Get training and testing for demonstrating the generative of the objectness over classes
    void getTrainTest();
    
public: // Used for testing the ability of generic over classes
    void loadDataGenericOverCls();
    
private:
    void loadBox(const cv::FileNode &fn, std::vector<cv::Vec4i> &boxes, vecI &clsIdx);
    bool loadBBoxes(CStr &nameNE, std::vector<cv::Vec4i> &boxes, vecI &clsIdx);
    static void getXmlStrVOC(CStr &fName, std::string &buf);
    static inline std::string keepXmlChar(CStr &str);
};

std::string DataSetVOC::keepXmlChar(CStr &_str)
{
    std::string str = _str;
    int sz = (int)str.size(), count = 0;
    for (int i = 0; i < sz; i++){
        char c = str[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ' ' || c == '.')
            str[count++] = str[i];
    }
    str.resize(count);
    return str;
}

double DataSetVOC::interUnio(const cv::Vec4i &bb, const cv::Vec4i &bbgt)
{
    int bi[4];
    bi[0] = std::max(bb[0], bbgt[0]);
    bi[1] = std::max(bb[1], bbgt[1]);
    bi[2] = std::min(bb[2], bbgt[2]);
    bi[3] = std::min(bb[3], bbgt[3]);
    
    double iw = bi[2] - bi[0] + 1;
    double ih = bi[3] - bi[1] + 1;
    double ov = 0;
    if (iw>0 && ih>0){
        double ua = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih;
        ov = iw*ih/ua;
    }	
    return ov;
}
#endif /* defined(__ImageTracking__DataSetVOC__) */
