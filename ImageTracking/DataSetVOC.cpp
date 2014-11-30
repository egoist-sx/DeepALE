//
//  DataSetVOC.cpp
//  ImageTracking
//
//  Created by bittnt on 29/11/2014.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//
#include "kyheader.h"

#include "DataSetVOC.h"


DataSetVOC::DataSetVOC(CStr &_wkDir)
{
    wkDir = _wkDir;
    resDir = wkDir + "model/";
    localDir = wkDir + "Local/";
    imgPathW = wkDir + "JPEGImages/%s.jpg";
    annoPathW = wkDir + "Annotations/%s.yml";
    
    trainNum = 0;
    testNum = 0;
}


cv::Vec4i getMaskRange(CMat &mask1u, int ext = 0)
{
    int maxX = INT_MIN, maxY = INT_MIN, minX = INT_MAX, minY = INT_MAX, rows = mask1u.rows, cols = mask1u.cols;
    for (int r = 0; r < rows; r++)	{
        const byte* data = mask1u.ptr<byte>(r);
        for (int c = 0; c < cols; c++)
            if (data[c] > 10) {
                maxX = std::max(maxX, c);
                minX = std::min(minX, c);
                maxY = std::max(maxY, r);
                minY = std::min(minY, r);
            }
    }
    
    maxX = maxX + ext + 1 < cols ? maxX + ext + 1 : cols;
    maxY = maxY + ext + 1 < rows ? maxY + ext + 1 : rows;
    minX = minX - ext > 0 ? minX - ext : 0;
    minY = minY - ext > 0 ? minY - ext : 0;
    
    return cv::Vec4i(minX + 1, minY + 1, maxX, maxY); // Rect(minX, minY, maxX - minX, maxY - minY);
}


DataSetVOC::~DataSetVOC(void)
{
}

void DataSetVOC::loadAnnotations()
{
    gtTrainBoxes.resize(trainNum);
    gtTrainClsIdx.resize(trainNum);
    for (int i = 0; i < trainNum; i++)
        if (!loadBBoxes(trainSet[i], gtTrainBoxes[i], gtTrainClsIdx[i]))
            return;
    
    gtTestBoxes.resize(testNum);
    gtTestClsIdx.resize(testNum);
    for (int i = 0; i < testNum; i++)
        if(!loadBBoxes(testSet[i], gtTestBoxes[i], gtTestClsIdx[i]))
            return;
    printf("Load annotations finished\n");
}

void DataSetVOC::loadDataGenericOverCls()
{
    vecS allSet = trainSet;
    allSet.insert(allSet.end(), testSet.begin(), testSet.end());
    int imgN = (int)allSet.size();
    trainSet.clear(), testSet.clear();
    trainSet.reserve(imgN), testSet.reserve(imgN);
    std::vector<std::vector<cv::Vec4i>> gtBoxes(imgN);
    std::vector<vecI> gtClsIdx(imgN);
    for (int i = 0; i < imgN; i++){
        if (!loadBBoxes(allSet[i], gtBoxes[i], gtClsIdx[i]))
            return;
        std::vector<cv::Vec4i> trainBoxes, testBoxes;
        vecI trainIdx, testIdx;
        for (size_t j = 0; j < gtBoxes[i].size(); j++)
            if (gtClsIdx[i][j] < 6){
                trainBoxes.push_back(gtBoxes[i][j]);
                trainIdx.push_back(gtClsIdx[i][j]);
            }
            else{
                testBoxes.push_back(gtBoxes[i][j]);
                testIdx.push_back(gtClsIdx[i][j]);
            }
        if (trainBoxes.size()){
            trainSet.push_back(allSet[i]);
            gtTrainBoxes.push_back(trainBoxes);
            gtTrainClsIdx.push_back(trainIdx);
        }
        else{
            testSet.push_back(allSet[i]);
            gtTestBoxes.push_back(testBoxes);
            gtTestClsIdx.push_back(testIdx);
        }
    }
    trainNum = 0;
    testNum =  0;
    printf("Load annotations (generic over classes) finished\n");
}

void DataSetVOC::loadBox(const cv::FileNode &fn, std::vector<cv::Vec4i> &boxes, vecI &clsIdx){
    std::string isDifficult;
    fn["difficult"]>>isDifficult;
    if (isDifficult == "1")
        return;
    
    std::string strXmin, strYmin, strXmax, strYmax;
    fn["bndbox"]["xmin"] >> strXmin;
    fn["bndbox"]["ymin"] >> strYmin;
    fn["bndbox"]["xmax"] >> strXmax;
    fn["bndbox"]["ymax"] >> strYmax;
    boxes.push_back(cv::Vec4i(atoi(_S(strXmin)), atoi(_S(strYmin)), atoi(_S(strXmax)), atoi(_S(strYmax))));
    
    std::string clsName;
    fn["name"]>>clsName;
    clsIdx.push_back(findFromList(clsName, classNames));
    
    //std::string error_message = "Invalidate class name\n";
    //CV_Assert_( (clsIdx[clsIdx.size() - 1] >= 0), error_message);
}

bool DataSetVOC::loadBBoxes(CStr &nameNE, std::vector<cv::Vec4i> &boxes, vecI &clsIdx)
{
    std::string fName = cv::format(_S(annoPathW), _S(nameNE));
    cv::FileStorage fs(fName, cv::FileStorage::READ);
    cv::FileNode fn = fs["annotation"]["object"];
    boxes.clear();
    clsIdx.clear();
    if (fn.isSeq()){
        for (cv::FileNodeIterator it = fn.begin(), it_end = fn.end(); it != it_end; it++){
            loadBox(*it, boxes, clsIdx);
        }
    }
    else
        loadBox(fn, boxes, clsIdx);
    return true;
}

// Get training and testing for demonstrating the generative of the objectness over classes
void DataSetVOC::getTrainTest()
{
    const int TRAIN_CLS_NUM = 6;
    std::string trainCls[TRAIN_CLS_NUM] = {"bird", "car", "cat", "cow", "dog", "sheep"};
    
}

void DataSetVOC::getXmlStrVOC(CStr &fName, std::string &buf)
{
    std::ifstream fin(fName);
    std::string strLine;
    buf.clear();
    buf.reserve(100000);
    buf += "<?xml version=\"1.0\"?>\n<opencv_storage>\n";
    while (getline(fin, strLine) && strLine.size())	{
        int startP = strLine.find_first_of(">") + 1;
        int endP = strLine.find_last_of("<");
        if (endP > startP){
            std::string val = keepXmlChar(strLine.substr(startP, endP - startP));
            if (val.size() < endP - startP)
                strLine = strLine.substr(0, startP) + val + strLine.substr(endP);
        }
        buf += strLine + "\n";
    }
    buf += "</opencv_storage>\n";
    //FileStorage fs(buf, FileStorage::READ + FileStorage::MEMORY);
    std::ofstream fout("D:/t.xml");
    fout<< buf;
}
