import os
import cv2
import numpy as np
import time
import argparse
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
def load_image(path):
    dimensions=[]   
    paths=[]
    if type(path)==list:
        paths=path
        blobs=[]
        for na,images in enumerate (path):
            i=cv2.imread(images)
            height,width,na=i.shape
            dimensions.append([width,height])
            input_blob = cv2.dnn.blobFromImage(i, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            blobs.append(input_blob)
    elif os.path.isfile(path):
        blobs=[]
        paths=[path]
        i=cv2.imread(path)
        height,width,na=i.shape
        input_blob = cv2.dnn.blobFromImage(i, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        blobs.append(input_blob)
        dimensions.append([width,height])
    else:
        blobs=[]
        for image_file in os.listdir(path):
            image_file=os.path.join(path,image_file)
            paths.append(image_file)
            i=cv2.imread(image_file)
            height,width,na=i.shape
            dimensions.append([width,height])
            input_blob = cv2.dnn.blobFromImage(i, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            blobs.append(input_blob)
    return blobs,dimensions,paths
def infer(nn,images,layers):
    tot=0
    check=0
    if len(images)>1:
        out=[]
        for image in (images):
            per=time.time()
            nn.setInput(image)
            output=nn.forward(layers)
            out.append(output)
            end=time.time()
            tot+=(end-per)
    else:
        per=time.time()
        nn.setInput(images[0])
        out=nn.forward(layers)
        end=time.time()   
        tot=(end-per)
    if len(images)>1:
        tot=tot/len(images)
    else:
        tot
    return out,tot
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
                if pc>0.7:
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
def get_info(re,classes,perimage,label,paths):
    lists=[]
    listt=[]
    cls=[]
    start=0
    for x,i in enumerate(re):
        path=paths[x]
        for l in range(perimage[x]):
           if(l in re[x]):
               listt.append(label[classes[l+start]])
               cls.append(label[classes[l+start]])
        lists.append([path,listt])
        listt=[]
        start+=perimage[x]
    unique=list(set(cls))
    numuni=len(unique)
    amnt=[]
    for obj in unique:
        amnt.append([obj,cls.count(obj)])

    return numuni, amnt, lists

def draw(classes,paths,bounding_boxes, confidences,perimage, probability_minimum, threshold,label):
    start=0
    end=0
    ind=[]
    for j, image in enumerate(paths):
        name=image
        image=cv2.imread(image)
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
                colour_box = [int(j) for j in colours[classes[start+i]]]
                cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                            colour_box, 5)
                text_box = '{}: {:.4f}'.format(label[classes[i+start]].strip(),confidences[start+i])
                cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)
        original,jpg=os.path.splitext(name)
        if"/"in original:
            file,imn=original.split("/")
        else:
            file,imn=original.split("\\")
        name=f"out_imgs/{imn}_out{jpg}"
        save=cv2.imwrite(name,image)
        print("done saving")   
        start=end 
    return ind 
         
def main():
    flag=argparse.ArgumentParser(description="To display inference time, total dected objects, object breakdown per image.")
    flag.add_argument("-inf", action="store_true", help="Average inference time.")
    flag.add_argument("-classes_all", action="store_true", help="Number of detected objects.")
    flag.add_argument("-Total_break", action="store_true", help="Detailed Number of detected objects.")
    flag.add_argument("-Per_breakdown", action="store_true", help="Detection breakdown per image.")
    flags=flag.parse_args()
    yolofiles, path=input("Please give Yolo file containing .names, .cfg, and weights. As well as 1 for image, 2 for image folder, 3 for select images ").split()
    path=int(path)
    saver=path
    if(path == 3):
        images=input("Please list all images paths with a comma in between: ").split(",")
        images=[image.strip() for image in images]
    elif(path ==1 or path ==2):
        images=input("Please give path to image/folder: ")
    label,config,weights,path=inputs(yolofiles,images)
    label=load_classes(label)
    nn,layers=load_model(weights,config)
    blobs,dimensions,paths=load_image(images)
    results,time=infer(nn,blobs,layers)
    print(time)
    classes,confidence,bb,perimage=detection(results,dimensions,saver)
    f=[]
    for i,l in enumerate(classes):
        f.append(label[l])
    re=draw(classes,paths,bb, confidence,perimage, probability_minimum=0.8, threshold=0.5,label=label)
    numuni,amnt,lists=get_info(re,classes,perimage,label,paths)
    if not(flags.inf or flags.classes_all or flags.Per_breakdown or flags.Total_break):
        flags.inf=True
        flags.classes_all=True
        flags.Per_breakdown=True
        flags.Total_break=True
    if flags.inf:
        print("Average Inference Time: {} seconds".format(time))
    if flags.classes_all:
        print("Total Number of Objects/Classes Detected: ",numuni)
    if flags.Total_break:
        print("Total Detection Breakdown:")
        for obj,num in amnt:
            print("{}: {}".format(obj,num))
    if flags.Per_breakdown:
         for x,(img, lis) in enumerate(lists):
            un=list(set(lis))
            print(img)
            for j in un:
                print("=>",j,lis.count(j))

            


if __name__=="__main__":
    main()








