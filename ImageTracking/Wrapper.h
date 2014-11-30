//
//  Wrapper.h
//  ImageTracking
//
//  Created by Xin Sun on 11/29/14.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "BBox.h"

@interface Wrapper : NSObject
+ (NSMutableArray*) getBBoxFromUIImage:(UIImage*) image;
+ (void) trainBing:(CGRect)rect;
@end
