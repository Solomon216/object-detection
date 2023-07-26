import cv2 , time , os , tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)

class Detector:
    def __init__(self):
        pass

    def readclasses(self,classfile):
        with open(classfile,'r') as f:
            self.classList=f.read().splitlines()
        self.colorList=np.random.uniform(0,255,size=(len(self.classList),3))

        print(len(self.colorList),len(self.classList))    

    def downloadmodel(self,modelurl):
        fileName=os.path.basename(modelurl)
        self.modelname=fileName[:fileName.find('.')]

        self.cachedir="./pretrained_models"
        os.makedirs(self.cachedir,exist_ok=True)

        get_file(fname=fileName,origin=modelurl,cache_dir=self.cachedir,cache_subdir="checkpoints",extract=True)

    def loadmodel(self):
        print("loading model"+self.modelname)
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cachedir,"checkpoints",self.modelname,"saved_model"))  
        print("model loaded")

    def createbox(self,image,threshold=0.5):
        input=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        input=tf.convert_to_tensor(input,dtype=tf.uint8)
        input=input[tf.newaxis,...]

        detections=self.model(input)

        boxes=detections['detection_boxes'][0].numpy()
        classindexes=detections['detection_classes'][0].numpy().astype(np.int32)
        scores=detections['detection_scores'][0].numpy()

        imgH,imgW,imgC=image.shape
        boxidx=tf.image.non_max_suppression(boxes,scores,max_output_size=50,iou_threshold=0.5,score_threshold=0.5)
        if len(boxidx)!= 0:
            for i in boxidx:
                box=tuple(boxes[i].tolist())
                classConf=round(100*scores[i])
                classIndex=classindexes[i]

                classname=self.classList[classIndex]
                classColor=self.colorList[classIndex]

                display='{}:{}%'.format(classname,classConf)

                ymin,xmin,ymax,xmax=box
                print(ymin,xmin,ymax,xmax)
                xmin,xmax,ymin,ymax =(xmin*imgW,xmax*imgW,ymin*imgH,ymax*imgH)
                xmin,xmax,ymin,ymax=int(xmin),int(xmax),int(ymin),int(ymax)
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),classColor,1)
                cv2.putText(image,display,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,classColor,2)
                linewidth=min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))
                cv2.line(image,(xmin,ymin),(xmin+linewidth,ymin),classColor,1)
                cv2.line(image,(xmin,ymin),(xmin,ymin+linewidth),classColor,1)

                cv2.line(image,(xmax,ymin),(xmax-linewidth,ymin),classColor,1)
                cv2.line(image,(xmax,ymin),(xmax,ymin+linewidth),classColor,1)

                cv2.line(image,(xmin,ymax),(xmin+linewidth,ymax),classColor,1)
                cv2.line(image,(xmin,ymax),(xmin,ymax+linewidth),classColor,1)

                cv2.line(image,(xmax,ymax),(xmax-linewidth,ymax),classColor,1)
                cv2.line(image,(xmax,ymax),(xmax,ymax+linewidth),classColor,1)
        return image

    def predimage(self,imgpath,threshold=0.5):
        image=cv2.imread(imgpath)

        boximage=self.createbox(image)

        cv2.imwrite(self.modelname+".jpg",boximage)
        cv2.imshow("image",boximage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predvideo(self,videoPath,threshold=0.5):
        cap=cv2.VideoCapture(videoPath)
        (success,image)=cap.read()

        starttime=0

        while success:
            currenttime=time.time()
            fps=1/(currenttime-starttime)
            starttime=currenttime
            boximage=self.createbox(image,threshold)
            cv2.putText(boximage,"fps:"+str(int(fps)),(20,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv2.imshow("image",boximage)
            key=cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                break
            (success,image)=cap.read()
            
        cv2.destroyAllWindows()