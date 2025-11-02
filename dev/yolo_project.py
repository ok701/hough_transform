import cv2
import numpy as np
import math

#Yolo 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#이미지 가져오기
#img = cv2.resize(img, (1080,724))


cap = cv2.VideoCapture('Los_Angeles_Trim_Trim.mp4')
if not cap.isOpened:
    print('unable to read camara feed')

count=0
running = True
while running:
    ret, img = cap.read()
    height, width, channels = img.shape
    if ret:
        y_end = height
        y_start = 3/5*y_end
        x_end = width


        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        equalized_gray = cv2.equalizeHist(gray)
        G_blur = cv2.GaussianBlur(equalized_gray,(5,5),0) # kernel_size = 5, sigmaX = 0
        edges = cv2.Canny(G_blur,100,200) # low_threshold = 100, high_threshold = 200

        roi_array = np.array([ [(x_end*1/5,y_end/3*4), (x_end*2/5,y_start), (x_end*3/5,y_start), (x_end*4/5,y_end)] ], dtype = np.int32) # 사다리꼴
        backgnd = np.zeros_like(edges)
        roi = cv2.fillPoly(backgnd,roi_array,(255,255,255)) # 흰색
        roi_edges = cv2.bitwise_and(edges,roi) # Canny ROI 논리 곱 연산


        H_lines = cv2.HoughLinesP(roi_edges,6,np.pi/60,90,40,25) # 원점부터 rho까지 분해능 = 6, 회전각도의 분해능 = pi/60


        x_right, y_right, x_left, y_left = [],[],[],[] # 초기화

        try:        
            for line in H_lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1)
                    if math.fabs(slope) < 1/2: # 기울기가 1/2 이하이면 무시한다
                        continue
                    if slope > 0: # 기울기가 양수이면           
                        x_right.extend([x1, x2])
                        y_right.extend([y1, y2])
                    else:           
                        x_left.extend([x1, x2])
                        y_left.extend([y1, y2])


            left_eq = np.poly1d(np.polyfit(y_left,x_left,1)) # 1차 함수로 근사
            right_eq = np.poly1d(np.polyfit(y_right,x_right,1))

            left_x_0 = int(left_eq(y_end))
            left_x_end = int(left_eq(y_start))
            right_x_0 = int(right_eq(y_end))
            right_x_end = int(right_eq(y_start))

            backgnd2 = np.zeros_like(img)
            new_lines = [[
                    [left_x_0, int(y_end), left_x_end, int(y_start)],
                    [right_x_0, int(y_end), right_x_end, int(y_start)],
                        ]]
            for line in new_lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(backgnd2,(x1,y1),(x2,y2),(0,0,255),10)

            backgnd2_prev=backgnd2
        except:
            pass


        count+=1
        if count%2==1:
            #물체감지
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

        #정보를 화면에 표시
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                cv2.putText(img, label, (x, y - 10), font, 1, (255,0,0), 2)

        try:
            res = cv2.addWeighted(backgnd2,1.,img,1.,0.)
        
        except:
            try:
                res = cv2.addWeighted(backgnd2_prev,1.,img,1.,0.)
            
            except:
                res= img
        cv2.imshow('Camara',res)
        #vid.update(res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc 버튼
        running = False

cap.release()
#vid.release()
cv2.destroyAllWindows()