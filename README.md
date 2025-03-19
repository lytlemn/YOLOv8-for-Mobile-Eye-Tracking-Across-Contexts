# YOLOv8-for-Mobile-Eye-Tracking-Across-Contexts
Implementation of YOLOv8 to identify areas of interest (AOIs) and intersections with gaze data acquired using Pupil Invisible Mobile Eye-Tracking (MET) system. We developed this project to process MET data for two ongoing projects in our lab. Code and sample data is divided by project as each has unique AOIs. However, code is largely similar across projects and individuals are encouraged to modify or reuse either version to best suit their study needs.

For more information about the project, samples, and AOIs associated with each project please see our associated presentations/publications:

## Sample MET Data and Walkthrough via Google Colab (/Tutorial)
In progress

## Validation Paper (/Val_Paper)
### Sample
To test the performance of our implementation of YOLO-v8 on detection of areas of interest across contexts we sampled participants from two existing mobile eye-tracking datasets examining participant social interactions. 10 young adult dyads  were sampled from the Behavioral Foundations of Friendship (BFFs) study, a study investigating peer interactions in young adult same gender friendships. 30 parent-child dyads were sampled from the Parent-to-Child Anxiety Transmission (PCAT) study. The PCAT study is a two-site accelerated longitudinal study designed to assess parent and child attentional, affective, and behavioral dynamics under stress which relate to change in children’s anxiety symptoms over time.
### /MET_processing
Pipelines for using YOLOv8 to process MET data for BFFs and PCAT and code to manually label one minute clips from each task
Model weights for the two YOLOv8 models we implemented can be found at https://docs.ultralytics.com/models/yolov8/#__tabbed_1_3 and  https://github.com/lindevs/yolov8-face
### /Data
Demographic and validation data for each project. Validation data contains only frames which gaze AOI was identified by both YOLOv8 and human labellers
### /Analyses
Validation analyses including descriptives, confusion matrices, and performance metrics by demographic characteristics
