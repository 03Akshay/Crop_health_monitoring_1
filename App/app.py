import os
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
#importing for weed or crop
import cv2
import matplotlib.pyplot as plt
import time

disease_info = pd.read_csv("Plant-Disease-Detection\\App\\disease_info.csv" , encoding='cp1252')
supplement_info = pd.read_csv("Plant-Disease-Detection\\App\\supplement_info.csv",encoding='cp1252')

model = CNN.CNN(13) 
model.load_state_dict(torch.load("Plant-Disease-Detection\\App\\plant_disease_model_2.pt", map_location=torch.device('cpu'), weights_only=True))

#model.load_state_dict(torch.load("Plant-Disease-Detection\\App\\plant_disease_model_2.pt", map_location=torch.device('cpu'),))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

def predict_weed(imge_path):
    labelsPath = r"Plant-Disease-Detection\\App\weed\\obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = r"Plant-Disease-Detection\\App\weed\\crop_weed_detection.weights"
    configPath = r"Plant-Disease-Detection\\App\weed\\crop_weed.cfg"
    #load weights and cfg
    
    #color selection for drawing bbox
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    print("[INFO]     : loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    

    #load our input image and grab its spatial dimensions
    image = cv2.imread(imge_path)
    (H, W) = image.shape[:2]

    #parameters
    confi = 0.5
    thresh = 0.5

    #determine only the *output* layer names that we need from YOLO
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    ln = [layer_names[i - 1] for i in out_layers.flatten()]

    #construct a blob from the input image and then perform a forward
    #pass of the YOLO object detector, giving us our bounding boxes and
    #associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    #show timing information on YOLO
    print("[INFO]     : YOLO took {:.6f} seconds".format(end - start))

    #initialize our lists of detected bounding boxes, confidences, and
    #class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    #loop over each of the layer outputs
    for output in layerOutputs:
        #loop over each of the detections
        for detection in output:
            #extract the class ID and confidence (i.e., probability) of
            #the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter out weak predictions by ensuring the detected
            #probability is greater than the minimum probability
            if confidence > confi:
                #scale the bounding box coordinates back relative to the
                #size of the image, keeping in mind that YOLO actually
                #returns the center (x, y)-coordinates of the bounding
                #box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                #use the center (x, y)-coordinates to derive the top and
                #and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                #update our list of bounding box coordinates, confidences,
                #and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #apply non-maxima suppression to suppress weak, overlapping bounding
    #boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
    print("[INFO]     : Detections done, drawing bounding boxes...")
    #ensure at least one detection exists
    if len(idxs) > 0:
        #loop over the indexes we are keeping
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            output=LABELS[classIDs[i]]
            acrcy=confidences[i]
            cv2.rectangle(image, (x, y), (x + w-100, y + h-100), color, 2)
            print("[OUTPUT]   : detected label -> ",output)
            print("[ACCURACY] : accuracy -> ", confidences[i])
            text = "{} : {:.4f}".format(LABELS[classIDs[i]] , confidences[i])
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.8, color, 2)
    else:
        acrcy=0
        output="we didn't determine whether is crop or weed"
    det = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #cv2.imshow('Output', det)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[STATUS]   : Completed")
    print("[END]")
    return output,acrcy

    
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        wd=predict_weed(file_path)
        print(wd)

        if wd[0]=="crop":
            pred = prediction(file_path)
            title = disease_info['disease_name'][pred]
            description =disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            print(supplement_info['supplement name'])

            if pred in supplement_info.index:
                supplement_name = supplement_info['supplement name'][pred]
                supplement_image_url = supplement_info['supplement image'][pred]
            else:
                print(f"Warning: {pred} is not a valid key in supplement_info.")
                supplement_name = "Unknown"
                supplement_image_url = "Unknown"
            accu=wd[-1]
            return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                                image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url,accu=accu )
        else:
            return render_template('weed.html',wd=wd)

            
       
@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']))


if __name__ == '__main__':
    app.run(debug=True)
