//
//  ViewController.m
//  ImageTracking
//
//  Created by Xin Sun on 11/29/14.
//  Copyright (c) 2014 Xin Sun. All rights reserved.
//

#import "ViewController.h"
#import "Wrapper.h"
#import <DeepBelief/DeepBelief.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <AVFoundation/AVFoundation.h>

@interface ViewController ()
@end

@implementation ViewController

BOOL swiped = NO;
CGPoint lastTouch;
CGPoint initialTouch;

float ox;
float oy;
float width;
float height;
BOOL flag = NO;


const unsigned char SpeechKitApplicationKey[] =

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setup];
   
    [SpeechKit setupWithID:
                      host:
                      port:443
                    useSSL:NO
                  delegate:nil];
    
    // Set earcons to play
    SKEarcon* earconStart	= [SKEarcon earconWithName:@"earcon_listening.wav"];
    SKEarcon* earconStop	= [SKEarcon earconWithName:@"earcon_done_listening.wav"];
    SKEarcon* earconCancel	= [SKEarcon earconWithName:@"earcon_cancel.wav"];
    
    [SpeechKit setEarcon:earconStart forType:SKStartRecordingEarconType];
    [SpeechKit setEarcon:earconStop forType:SKStopRecordingEarconType];
    [SpeechKit setEarcon:earconCancel forType:SKCancelRecordingEarconType];
    
 
}

- (void)setup {
    [_boxView setBackgroundColor:[UIColor clearColor]];
    
    
//    [self drawRectangle:CGRectMake(149, 293, 31, 15)];
    
//    [self drawRectangle:_imageView.frame];
//    NSLog(@"ox: %f, oy: %f, width: %f, height: %f", ox,oy,width, height);
}

- (CVPixelBufferRef) pixelBufferFromCGImage: (CGImageRef) image
{
    
    CGSize frameSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithBool:NO], kCVPixelBufferCGImageCompatibilityKey,
                             [NSNumber numberWithBool:NO], kCVPixelBufferCGBitmapContextCompatibilityKey,
                             nil];
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width,
                                          frameSize.height,  kCVPixelFormatType_32ARGB, (__bridge CFDictionaryRef) options,
                                          &pxbuffer);
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);
    
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    
    
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, frameSize.width,
                                                 frameSize.height, 8, CVPixelBufferGetBytesPerRow(pxbuffer), rgbColorSpace,
                                                 kCGImageAlphaNoneSkipLast);
    
    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image),
                                           CGImageGetHeight(image)), image);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    
    return pxbuffer;
}

- (UIImage *)imageWithImage:(UIImage *)image scaledToSize:(CGSize)newSize {
    //UIGraphicsBeginImageContext(newSize);
    // In next line, pass 0.0 to use the current device's pixel scaling factor (and thus account for Retina resolution).
    // Pass 1.0 to force exact pixel size.
    UIGraphicsBeginImageContextWithOptions(newSize, NO, 0.0);
    [image drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}

- (void)runCNNOnFrame
{

    void* inputImage = jpcnn_create_image_buffer_from_file([imagePath UTF8String]);
    float* predictions;
    int predictionsLength;
    char** predictionsLabels;
    int predictionsLabelsLength;
    
    NSString* networkPath = [[NSBundle mainBundle] pathForResource:@"jetpac" ofType:@"ntwk"];
    if (networkPath == NULL) {
        fprintf(stderr, "Couldn't find the neural network parameters file - did you add it as a resource to your application?\n");
        assert(false);
    }
    network = jpcnn_create_network([networkPath UTF8String]);
    assert(network != NULL);
    
    jpcnn_classify_image(network, inputImage, 0, 0, &predictions, &predictionsLength, &predictionsLabels, &predictionsLabelsLength);
    
    
    jpcnn_destroy_image_buffer(inputImage);
    float highest = -1;
    int index2 = -1;
    for (int index = 0; index < predictionsLength; index += 1) {
        const float predictionValue = predictions[index];
        if (predictionValue > 0.01) {
            if(predictionValue > highest) {
                highest = predictionValue;
                index2 = index;
            }
            char* label = predictionsLabels[index % predictionsLabelsLength];
            NSString* labelObject = [NSString stringWithCString: label];
            NSNumber* valueObject = [NSNumber numberWithFloat: predictionValue];
            NSLog(@"Label: %@, confidence: %f", labelObject, predictionValue);
        }
    }
    char* label = predictionsLabels[index2];
    NSString* most = [NSString stringWithCString:label];
//    _textLabel.text = most;
}

//- (void) setPredictionValues: (NSDictionary*) newValues {
//    const float decayValue = 0.75f;
//    const float updateValue = 0.25f;
//    const float minimumThreshold = 0.01f;
//    
//    for (NSString* label in newValues) {
//        NSNumber* newPredictionValueObject =
//    }
//    
//    NSMutableDictionary* decayedPredictionValues = [[NSMutableDictionary alloc] init];
//    for (NSString* label in oldPredictionValues) {
//        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
//        const float oldPredictionValue = [oldPredictionValueObject floatValue];
//        const float decayedPredictionValue = (oldPredictionValue * decayValue);
//        if (decayedPredictionValue > minimumThreshold) {
//            NSNumber* decayedPredictionValueObject = [NSNumber numberWithFloat: decayedPredictionValue];
//            [decayedPredictionValues setObject: decayedPredictionValueObject forKey:label];
//        }
//    }
//    [oldPredictionValues release];
//    oldPredictionValues = decayedPredictionValues;
//    
//    for (NSString* label in newValues) {
//        NSNumber* newPredictionValueObject = [newValues objectForKey:label];
//        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
//        if (!oldPredictionValueObject) {
//            oldPredictionValueObject = [NSNumber numberWithFloat: 0.0f];
//        }
//        const float newPredictionValue = [newPredictionValueObject floatValue];
//        const float oldPredictionValue = [oldPredictionValueObject floatValue];
//        const float updatedPredictionValue = (oldPredictionValue + (newPredictionValue * updateValue));
//        NSNumber* updatedPredictionValueObject = [NSNumber numberWithFloat: updatedPredictionValue];
//        [oldPredictionValues setObject: updatedPredictionValueObject forKey:label];
//    }
//    NSArray* candidateLabels = [NSMutableArray array];
//    for (NSString* label in oldPredictionValues) {
//        NSNumber* oldPredictionValueObject = [oldPredictionValues objectForKey:label];
//        const float oldPredictionValue = [oldPredictionValueObject floatValue];
//        if (oldPredictionValue > 0.05f) {
//            NSDictionary *entry = @{
//                                    @"label" : label,
//                                    @"value" : oldPredictionValueObject
//                                    };
//            candidateLabels = [candidateLabels arrayByAddingObject: entry];
//        }
//    }
//    NSSortDescriptor *sort = [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
//    NSArray* sortedLabels = [candidateLabels sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];
//    
//    
//    for (NSDictionary* entry in sortedLabels) {
//        NSString* label = [entry objectForKey: @"label"];
//        NSNumber* valueObject =[entry objectForKey: @"value"];
//        const float value = [valueObject floatValue];
//        
//    }
//}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)takePhoto:(id)sender {
    UIImagePickerController *picker = [[UIImagePickerController alloc] init];
    picker.delegate = self;
    picker.allowsEditing = FALSE;
    picker.sourceType = UIImagePickerControllerSourceTypeCamera;
    [self presentViewController:picker animated:YES completion:NULL];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary *)info {
    
    UIImage *chosenImage = info[UIImagePickerControllerOriginalImage];
    UIImage* resizeImage = [self imageWithImage:chosenImage scaledToSize:CGSizeMake(350, 467)];
    NSLog(@"Size: %f, %f", chosenImage.size.width, chosenImage.size.height);
    self.imageView.image = chosenImage;
   
    NSData *pngData = UIImagePNGRepresentation(resizeImage);
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsPath = [paths objectAtIndex:0]; //Get the docs directory
    NSString *filePath = [documentsPath stringByAppendingPathComponent:@"image.png"]; //Add the file name
    imagePath = filePath;
    [pngData writeToFile:filePath atomically:YES]; //Write the file
    UIGraphicsBeginImageContext(self.view.frame.size);
    _boxView.image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();

    [self runCNNOnFrame];
    
    
    [picker dismissViewControllerAnimated:YES completion:NULL];
}

- (void) drawRectangle:(CGRect) rect {
    UIGraphicsBeginImageContext(_boxView.frame.size);
    UIColor * redColor = [UIColor colorWithRed:1.0 green:0.0 blue:0.0 alpha:1.0];
    
    CGContextSetLineWidth(UIGraphicsGetCurrentContext(), 2);
    CGContextMoveToPoint(UIGraphicsGetCurrentContext(), rect.origin.x, rect.origin.y);
    CGContextAddLineToPoint(UIGraphicsGetCurrentContext(), rect.origin.x+rect.size.width, rect.origin.y);
    
    
    CGContextMoveToPoint(UIGraphicsGetCurrentContext(), rect.origin.x, rect.origin.y);
    CGContextAddLineToPoint(UIGraphicsGetCurrentContext(), rect.origin.x, rect.origin.y+rect.size.height);
    CGContextMoveToPoint(UIGraphicsGetCurrentContext(), rect.origin.x+rect.size.width, rect.origin.y+rect.size.height);
    CGContextAddLineToPoint(UIGraphicsGetCurrentContext(), rect.origin.x+rect.size.width, rect.origin.y);
    CGContextMoveToPoint(UIGraphicsGetCurrentContext(), rect.origin.x+rect.size.width, rect.origin.y+rect.size.height);
    CGContextAddLineToPoint(UIGraphicsGetCurrentContext(), rect.origin.x, rect.origin.y+rect.size.height);
    
   CGContextSetStrokeColorWithColor(UIGraphicsGetCurrentContext(), redColor.CGColor);
    CGContextStrokePath(UIGraphicsGetCurrentContext());
    
    UIImage* newImage = UIGraphicsGetImageFromCurrentImageContext();
    _boxView.image = newImage;
    UIGraphicsEndImageContext();
}

-(CGRect) transformedLocation:(CGRect) origin {
    float dx=_imageView.frame.origin.x;
    float dy=_imageView.frame.origin.y;
    return CGRectMake(origin.origin.x+dx, origin.origin.y+dy, origin.size.width, origin.size.height);
}

- (void) drawBBoxOfImage:(UIImage*) image {
    UIGraphicsBeginImageContext(self.view.frame.size);
    _boxView.image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    NSMutableArray* array = [Wrapper getBBoxFromUIImage:image];
    for (int i=0;i < [array count]; i++) {
        BBox* box = array[i];
        CGRect rect = [self transformedLocation:[box getCGRect]];
        NSLog(@"ox: %f, oy: %f, width: %f, height: %f", rect.origin.x, rect.origin.y, rect.size.width, rect.size.height);
        [self drawRectangle:rect];
    }
    
}

- (IBAction)getVoice:(id)sender {
    self.speechBtn.selected = !self.speechBtn.isSelected;
    
    // This will initialize a new speech recognizer instance
    if (self.speechBtn.isSelected) {
        
        self.voiceSearch = [[SKRecognizer alloc] initWithType:SKSearchRecognizerType
                                                    detection:SKShortEndOfSpeechDetection
                                                     language:@"en_US"
                                                     delegate:self];
    }
    
    // This will stop existing speech recognizer processes
    else {
        if (self.voiceSearch) {
            [self.voiceSearch stopRecording];
            [self.voiceSearch cancel];
        }
    }
}

- (void)recognizerDidBeginRecording:(SKRecognizer *)recognizer {
}

- (void)recognizerDidFinishRecording:(SKRecognizer *)recognizer {
}

- (void)recognizer:(SKRecognizer *)recognizer didFinishWithResults:(SKRecognition *)results {
    long numOfResults = [results.results count];
    
    if (numOfResults > 0) {
        // update the text of text field with best result from SpeechKit
        NSString* result = [results firstResult];
        _textLabel.text = result;
        NSArray* words = [result componentsSeparatedByString:@" "];
        for(int i = 0; i < [words count]; i++) {
            NSString* word = words[i];
            NSString* formatted = [word lowercaseString];
            NSArray* keywords = @[@"keyboard", @"cup", @"mug", @"monitor", @"car", @"laptop"];
            if ([keywords containsObject:formatted]) {
                NSLog(@"found: %@", formatted);
                __keyword.text = formatted;
            }
            UIGraphicsBeginImageContext(self.view.frame.size);
            _boxView.image = UIGraphicsGetImageFromCurrentImageContext();
            UIGraphicsEndImageContext();

            [self drawRectangle:CGRectMake(initialTouch.x, initialTouch.y, lastTouch.x-initialTouch.x, lastTouch.y-initialTouch.y)];
        }
    }
    
    self.speechBtn.selected = !self.speechBtn.isSelected;
    
    if (self.voiceSearch) {
        [self.voiceSearch cancel];
    }
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event {
    if (!flag) {
        swiped = NO;
        UITouch* touch = [touches anyObject];
        initialTouch = [touch locationInView:self.view];
        lastTouch = initialTouch;
    }
    
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event {
    if (!flag) {
        swiped = YES;
        UITouch *touch = [touches anyObject];
        CGPoint currentPoint = [touch locationInView: self.view];
        
        UIGraphicsBeginImageContext(self.view.frame.size);
        [_boxView.image drawInRect:CGRectMake(0, 0, self.view.frame.size.width, self.view.frame.size.height)];
        CGContextMoveToPoint(UIGraphicsGetCurrentContext(), lastTouch.x, lastTouch.y);
        CGContextAddLineToPoint(UIGraphicsGetCurrentContext(), currentPoint.x, currentPoint.y);
        CGContextSetLineWidth(UIGraphicsGetCurrentContext(), 10);
        CGContextSetRGBStrokeColor(UIGraphicsGetCurrentContext(), 250, 50, 50, 0.7);
        CGContextStrokePath(UIGraphicsGetCurrentContext());
        _boxView.image = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
        
        lastTouch = currentPoint;

    }
}

- (void) touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event {
    
    //clear image buffer
    if (abs(lastTouch.x-initialTouch.x)+abs(lastTouch.y-initialTouch.y) > 100) {
        //count, draw box;
        flag = YES;
        
//        [Wrapper trainBing:CGRectMake(initialTouch.x, initialTouch.y, lastTouch.x, lastTouch.y)];
//        [self drawRectangle:CGRectMake(initialTouch.x, initialTouch.y, lastTouch.x-initialTouch.x, lastTouch.y-initialTouch.y)];
//        [[UIApplication sharedApplication] beginIgnoringInteractionEvents];
    } else {
        UIGraphicsBeginImageContext(self.view.frame.size);
        _boxView.image = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();
    }
}

//- (void)runCNNOnFrame: (CVPixelBufferRef) pixelBuffer
//{
//    assert(pixelBuffer != NULL);
//    
//    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType( pixelBuffer );
//    int doReverseChannels;
//    if ( kCVPixelFormatType_32ARGB == sourcePixelFormat ) {
//        doReverseChannels = 1;
//    } else if ( kCVPixelFormatType_32BGRA == sourcePixelFormat ) {
//        doReverseChannels = 0;
//    } else {
//        assert(false); // Unknown source format
//    }
//    
//    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow( pixelBuffer );
//    const int width = (int)CVPixelBufferGetWidth( pixelBuffer );
//    const int fullHeight = (int)CVPixelBufferGetHeight( pixelBuffer );
//    CVPixelBufferLockBaseAddress( pixelBuffer, 0 );
//    unsigned char* sourceBaseAddr = CVPixelBufferGetBaseAddress( pixelBuffer );
//    int height;
//    unsigned char* sourceStartAddr;
//    if (fullHeight <= width) {
//        height = fullHeight;
//        sourceStartAddr = sourceBaseAddr;
//    } else {
//        height = width;
//        const int marginY = ((fullHeight - width) / 2);
//        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
//    }
//    void* cnnInput = jpcnn_create_image_buffer_from_uint8_data(sourceStartAddr, width, height, 4, sourceRowBytes, doReverseChannels, 1);
//    float* predictions;
//    int predictionsLength;
//    char** predictionsLabels;
//    int predictionsLabelsLength;
//    
//    struct timeval start;
//    gettimeofday(&start, NULL);
//    jpcnn_classify_image(network, cnnInput, JPCNN_RANDOM_SAMPLE, -2, &predictions, &predictionsLength, &predictionsLabels, &predictionsLabelsLength);
//    struct timeval end;
//    gettimeofday(&end, NULL);
//    const long seconds  = end.tv_sec  - start.tv_sec;
//    const long useconds = end.tv_usec - start.tv_usec;
//    const float duration = ((seconds) * 1000 + useconds/1000.0) + 0.5;
//    //  NSLog(@"Took %f ms", duration);
//    
//    jpcnn_destroy_image_buffer(cnnInput);
//    
//    dispatch_async(dispatch_get_main_queue(), ^(void) {
//        [self handleNetworkPredictions: predictions withLength: predictionsLength];
//    });
//}

@end
