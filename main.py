from detector import *

classfile="coco.names"
detector=Detector()
detector.readclasses(classfile)
imagepath="D:\\nearby\images (5) (2).jpeg"
videopath=0
threshold=0.5
#modelurl="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
modelurl="http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#modelurl="http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz"
detector.downloadmodel(modelurl)
detector.loadmodel()
detector.predimage(imagepath,threshold)   
#detector.predvideo(videopath,threshold)