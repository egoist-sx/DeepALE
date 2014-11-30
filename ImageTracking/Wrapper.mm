//
//  Wrapper.m
//  ImageTracking
//
//  Created by Xin Sun on 11/29/14.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#import "Wrapper.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "UIImage+OpenCV.h"
#include "BBox.h"

#include "DataSetVOC.h"
#include "Objectness.h"
#include "FilterTIG.h"
#include "kyheader.h"
#include "ValStructVec.h"
@implementation Wrapper

CGRect _rect;

void getBoxesfromBing(cv::Mat &image, std::vector<cv::Vec4i> &boxes);

+ (NSMutableArray*) getBBoxFromUIImage:(UIImage*) image {
    NSMutableArray* array = [[NSMutableArray alloc] init];
    cv::Mat img = image.CVMat3;
    std::vector<cv::Vec4i> boxes;
    getBoxesfromBing(img, boxes);
    for(int i=0;i<boxes.size();i++)
    {
        cv::Vec4i vec = boxes[i];
        BBox* box = [[BBox alloc] initWithCGRect:CGRectMake(vec[0], vec[1], vec[2]-vec[0], vec[3]-vec[1])];
        [array addObject:box];
    }
    return array;
}

+ (void) trainBing:(CGRect)rect {
    _rect = rect;
//    cv::Vec4i vec;
//    vec[0] = rect.origin.x;
//    vec[1] = rect.origin.y;
//    vec[2] = rect.size.width;
//    vec[3] = rect.size.height;
}

@end

void getBoxesfromBing(cv::Mat &image, std::vector<cv::Vec4i> &boxes){
    int widthofimage, heightofimage;
    widthofimage = image.cols;
    heightofimage= image.rows;
    
    
    NSString* model_idx = [[NSBundle mainBundle] pathForResource:@"ObjNessB2W8MAXBGR" ofType:@"idx"];
    
    //NSString *foo = @"Foo";
    //    std::string *model_idx_string = new std::string([model_idx UTF8String]);
    //    std::string *model_wS1_string = new std::string([model_wS1 UTF8String]);
    //    std::string *model_wS2_string = new std::string([model_wS2 UTF8String]);
    std::string model_idx_string([model_idx UTF8String]);
    
    
    cv::Mat query = image;
    
    cv::resize(query, query, cv::Size(350,467));
    cv::cvtColor(query, query, CV_BGR2RGB);
    
    
    srand((unsigned int)time(NULL));
    DataSetVOC voc2007("");
    double base = 2;
    int W = 8;
    int NSS = 2;
    int numPerSz = 130;
    
    Objectness objNess(voc2007,model_idx_string, base, W, NSS);
    
    ValStructVec<float, cv::Vec4i> bboxes;
    
    objNess.loadTrainedModel();
    objNess.getObjBndBoxes(query, bboxes, numPerSz);
    
    
    /*
     for (int i = 0; i < bboxes.size();i++){
     Rect current(cv::Point2i(bboxes[i][0],bboxes[i][1]),Point2i(bboxes[i][2],bboxes[i][3]));
     cv::rectangle(query,current,cv::Scalar(255,0,0),3);
     cv::imshow("query",query);
     cv::waitKey();
     }
     */
  
    boxes.reserve(1);
    cv::Vec4i box;
    box[0] = _rect.origin.x;
    box[1] = _rect.origin.y;
    box[2] = _rect.size.width;
    box[3] = _rect.size.height;
    boxes.push_back(box);
//    boxes.reserve(2);
//    for (int i = 0; i < 2; i++) {
//        cv::Vec4i box;
//        box[0] = std::max(bboxes[0][0],0);
//        box[1] = std::max(bboxes[0][1],0);
//        box[2] = std::min(bboxes[0][2],widthofimage);
//        box[3] = std::min(bboxes[0][3],heightofimage);
//        boxes.push_back(box);
//    }
}