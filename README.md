# YOLO_object_detection_for_ocr

This repository is work in progress!

## To Do

* ~~Refactor Code~~
* Write doc strings
* Design unit tests
* implement logging
* Write README file
* Enable IoU function to work with tensors
* implement mAP as performance metric in the model training: https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
* Fix plot function to work without list of bounding boxes
* New dataset (bigger and better)
* Determine and use best anchor boxes to have specialized predictors
* Adjust LearningRateScheduler according to YOLO paper: 10e-03, 10e-02 * 75 epochs, 10e-03 * 30 epochs, 10e-04 * 30 epochs
