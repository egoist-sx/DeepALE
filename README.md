DeepALE
============
by Xin Sun, Shuai Zheng, 30th Nov, 2014.
Deep ConvNet powered Automatic Labelling Environment helps to create tags on the objects in images.
============
We are motived by the recent progresses in Image-Recognition Software from Google/Stanford, which allow computer to automatic generate a sentence to describe the image. This type of technologies allow us to have better way to catalog and search for the billions of images and hours of video available online. As the increase of availability of images and videos, it becomes necessary to have such a tool to allow users and other third-parties software to find the objects of interest in image/video easier.

We develop an iOS app that allows users to tag nonuse for the objects of interest (e.g. chair, monitor, mug, keyboard, laptop, mouse etc) from favourite images/videos by just talking/finger-touching the image presented in the app. This app provides the function to recognise the query sentence from users, and recognise the objects being mentioned in the sentence, and highlight the objects in the image with bounding box.

The target users are mainly the users of smart phone and smart glass/smart watch. One target user group would be the partial sighted people, who would be relying on the new smart glass to navigate the scene and find their objects of interest.

This software  recognise the sentence from users via Nauance Dragon mobile SDKs. Then the software would automatically recognise the nounces in the sentence, and try to tag the related objects of interest in the images and videos. The recognition part is powered by BING software and the DeepBeliefSDK (Google API).

##Please note that currently bounding box is pre-drawed by user, which further work should be done to integrate either sliding window or BING to propose bounding boxes, then feed each segment to cnn for classification.

Please also supply your own key for voice recognition SDK, which could be applied here http://nuancemobiledeveloper.com/
