//
//  BBox.h
//  ImageTracking
//
//  Created by Xin Sun on 11/29/14.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@interface BBox : NSObject

@property CGRect rect;
@property float score;
- (CGRect) getCGRect;
- (BBox*) initWithCGRect:(CGRect) rect;
@end
