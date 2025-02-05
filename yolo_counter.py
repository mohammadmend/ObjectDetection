import os
import cv2
from sort import *
def inputs(yolo_files,path):
        for option in os.listdir(yolo_files):
            if option.endswith(".cfg"):
                config=os.path.join(yolo_files,option)
            elif option.endswith(".names"):
                label=os.path.join(yolo_files,option)
            else:
                weights=os.path.join(yolo_files,option)
        path=path
        return label,config,weights,path

    
def load_classes(path):
    names=[]
    if os.path.isfile(path):
        for idx,l in enumerate(open(path,"r").readlines()):
            names.append(l)
    return names
def load_model(weightp, configf):
    network = cv2.dnn.readNetFromDarknet(configf, weightp)
    layers = network.getLayerNames()
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']    
    return network,yolo_layers
def load_video_helper(path):
    blobs=[]
    dim=[]
    i=cv2.VideoCapture(path)
    while True:
        na,fram=i.read()
        if not na:
            break
        h,w,n=fram.shape
        dim.append([w,h])
        blob=cv2.dnn.blobFromImage(fram, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        blobs.append(blob)
    i.release()
    num_frame=len(blobs)
    return blobs,dim,num_frame
def load_video(path):
    paths=[]
    blobs=[]
    paths=[path]
    blob,dim,frame=load_video_helper(path)
    blobs.append(blob)
 

    return blobs,dim,paths,frame
def infer(nn,videos,layers,frames):
    tot=0
    t=[]
    out=[]
    for idx, blob in enumerate(videos):
        for blo in blob:
            per=time.time()
            nn.setInput(blo)
            output=nn.forward(layers)
            out.append(output)
            end=time.time()
            t.append(end-per)
            tot+=(end-per)
    tot=tot/(frames)
    return out,tot,t
def detection(results,dimensions,num):
    classes=[]
    confidence=[]
    bb=[]
    perimage=[]
    counter1=0
    if num==1:
        results=[results]
    for x,result in enumerate(results):
        counter=0
        for objects in result:
            for detection in objects:
                scores=detection[5:]
                detect=np.argmax(scores)
                pc=scores[detect]
                if pc>0.25:
                    iw, ih = dimensions[counter1] 
                    bc=detection[0:4]*np.array([iw,ih,iw,ih])
                    x, y, w, h = bc.astype('int')
                    x_min = int(x-(w/2))
                    y_min = int(y-(h/2))
                    bb.append([x_min,y_min,int(w),int(h)])
                    confidence.append(pc)
                    classes.append(detect)
                    counter+=1
        perimage.append(counter)
        counter1+=1
    return classes,confidence,bb,perimage

def draw(classes,paths,bounding_boxes, confidences,perimage, probability_minimum, threshold,label,dimensions,time):
    start=0
    end=0
    ind=[]
    tracker=Sort(max_age=1,min_hits=1)
    mot=[]
    person={}
    count=1
    boxes=[]
    pathsn=paths[0].replace("-raw","-result")
    cap=cv2.VideoCapture(paths[0])
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(pathsn,fourcc,20.0,(dimensions[0][0],dimensions[0][1]))
    j=0
    while cap.isOpened():
        ret,frame=cap.read()
        if j>= len(perimage):
            break
        end=perimage[j]+start
        results = cv2.dnn.NMSBoxes(bounding_boxes[start:end], confidences[start:end], probability_minimum, threshold)
        ind.append(results)
        coco_labels = 80
        np.random.seed(42)
        colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[start+i][0], bounding_boxes[start+i][1]
                box_width, box_height = bounding_boxes[start+i][2], bounding_boxes[start+i][3]
                if classes[i+start]==0:
                    boxes.append([x_min, y_min, box_width+x_min, y_min+box_height,confidences[start + i]])
        Alltrack=tracker.update(np.array(boxes))
        for track in Alltrack:
            x_min,y_min,x,y,id=track
            if id not in person:
                person[id]=count
                count+=1
            colour_box = (255,255,0)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x), int(y)),
                        colour_box, 5)
            text_box = '{}'.format(person[id])
            text_box1 = 'People: {}'.format(len(person))
            text_box2='Time of frame: {:.2f}'.format(time[j])

            cv2.putText(frame, text_box, (int(x_min), int(y_min) - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)
            cv2.putText(frame,text_box1,(int(dimensions[0][0])//2, int(dimensions[0][1]) - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)
            cv2.putText(frame,text_box2,(int(dimensions[0][0])//2, (int(dimensions[0][1])-int(dimensions[0][1])//2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)

        out.write(frame)
        mot.append({"fr":j,"box":boxes})
        boxes=[]
        start=end 
        j+=1
    cap.release()
    out.release()
    return ind,count,person
def main():
    yolofiles, path=input("Please give Yolo file containing .names, .cfg, and weights. As well as video path. ").split()
    path=(path)
    saver=path
    label,config,weights,path=inputs(yolofiles,path)
    label=load_classes(label)
    nn,layers=load_model(weights,config)
    blobs,dimensions,paths,frame=load_video(path)
    results,time,t=infer(nn,blobs,layers,frame)
    print(time)
    classes,confidence,bb,perimage=detection(results,dimensions,saver)
    re,count,person=draw(classes,paths,bb, confidence,perimage, probability_minimum=0.5, threshold=0.3,label=label,dimensions=dimensions,time=t)
if __name__=="__main__":
    main()








