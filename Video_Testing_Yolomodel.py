import cv2
import numpy as np
import glob

# Assigning required variable
d=0
img_count=0
sec=0.0
frameRate=1.0
coil_count=0
coupling_count=0
code_count=0
cc=0
cp=0

# Assigning Yolo model weight and config file
net = cv2.dnn.readNet("/home/vert/OCR/yolov3-config/yolov3_training_last_10.weights", "/home/vert/OCR/yolov3-config/yolov3_testing.cfg")

# Declaring classes
classes = []
with open("/home/vert/OCR/yolov3-config/classes.txt", "r") as f:
    classes = f.read().splitlines()

# Reading the Video from local machine
cap = cv2.VideoCapture("/home/vert/OCR/Video/video_cctv_WagonSideView1_2020-12-08 07_41_59.101766.avi")

# Assigning font style and color
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

# Running While loop in the video to testing our model
while True:
    try:
        sec = sec + frameRate
        sec = round(sec, 2)

        # Assigning Original image
        _, img = cap.read()
        if not img.data:
            print(img.data)

        # If image is coming blank need to continue with our exit
        if img is None:
            continue

        # How to take image per sec
        if sec %10 != 0:
            continue
        # print("sec", sec)

        # Assigning height and width to image
        # height, width, _ = img.shape
        # size=(width,height)

        # Scaling image to display while running code.
        scale_percent1 = 40
        width = int(img.shape[1] * scale_percent1 / 50)
        height = int(img.shape[0] * scale_percent1 / 50)

        # Editing original images according to our requirement
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        height, width, _ = img.shape
        size = (width, height)

        # Note that we are scaling image to 40% just to display it (img_to_display)...
        # while saving video we are using original image (img)

        # Converting image into blob to give yolo mode
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        # Assigning blank list
        boxes = []
        confidences = []
        class_ids = []

        # Running for loop to get the confidence score for images
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # According to my required i need fix the confidence score.
                if confidence > 0.10:

                    # Creating x y codinate to get the bounding box
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    # Append boxes list
                    boxes.append([x, y, w, h])

                    # Append confidences list
                    confidences.append((float(confidence)))

                    # Append ClassID list
                    class_ids.append(class_id)

        # Finally result will store in indexes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        # After completing above process we will get some value in indexes. Need to check the class
        if len(indexes)>0:
            for i in indexes.flatten():

                # bounding box
                x, y, w, h = boxes[i]

                # get the label from the class
                label = str(classes[class_ids[i]])
                #confidence = float(round(confidences[i],2))
                #color = colors[i]
                #cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)

                # writing the letter in video
                cv2.putText(img, label + " " + str(confidence), (x,y+20), font, 2, (255,255,255), 2)
                #cv2.putText(img, "*", (10,20), font, 2, (0,0,0), 2)
                #cv2.putText(img, "*", (740, 20), font, 2, (0, 0, 0), 2)
                #c1, c2 = (x + w) / 2, (y + h) / 2

                coil_count+=1
                # p1 = np.array((c1,c2))
                # p2 = np.array((370,20))
                # d = np.linalg.norm(p1-p2)

            # After all the process if we get code from the video need to increasing the count
                if "code" == classes[class_ids[i]]:
                    cc+=1
                    if cc==2:
                        code_count+=1

                # If object is coupling then need to take the count
                if "coupling" == classes[class_ids[i]]:
                    cp+=1
                    if cp==2:
                        coupling_count+=1
                        #cc=0
                        print("Coupling Count - {}".format(coupling_count))
                        print("Code Count - {}".format(code_count))

                # If object is wagon need to do below task
                if "wagon" == classes[class_ids[i]] and len(indexes)==1:
                    cp=0

        cv2.imshow("Frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Coupling Count - {}".format(coupling_count))
            print("Code Count - {}".format(code_count))
    except Exception as e:
        print("Theres is exception-{}".format(e))

cap.release()
cv2.destroyAllWindows()