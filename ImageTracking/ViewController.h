//
//  ViewController.h
//  ImageTracking
//
//  Created by Xin Sun on 11/29/14.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "Wrapper.h"
#import "BBox.h"
#import <SpeechKit/SpeechKit.h>
#import "AppDelegate.h"

@interface ViewController : UIViewController<UIImagePickerControllerDelegate, UINavigationControllerDelegate, SpeechKitDelegate,SKRecognizerDelegate> {
    
    SKRecognizer* voiceSearch;
    void* network;
    
    NSString* imagePath;
}

@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) IBOutlet UIButton *speechBtn;
@property (strong, nonatomic) IBOutlet UIButton *photoBtn;
@property (strong, nonatomic) IBOutlet UIImageView *boxView;
@property (strong, nonatomic) IBOutlet UILabel *textLabel;
@property (strong, nonatomic) SKRecognizer* voiceSearch;
@property (strong, nonatomic) IBOutlet UILabel *_keyword;
- (IBAction)takePhoto:(UIButton*)sender;
- (IBAction)getVoice:(UIButton*)sender;
@end

