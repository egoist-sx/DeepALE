//
//  BBox.m
//  ImageTracking
//
//  Created by Xin Sun on 11/29/14.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#import "BBox.h"

@implementation BBox
- (BBox*) initWithCGRect:(CGRect) rect {
    _rect = rect;
    return self;
}
- (CGRect) getCGRect {
    return _rect;
}
@end
