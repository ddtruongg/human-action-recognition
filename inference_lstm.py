import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
normal_threshold = 0.75

label = "........." 
n_time_steps = 10 # (timestep) tạo thành một chuỗi dữ liệu đặc trưng.
lm_list = [] # Danh sách các điểm đặc trưng trên cơ thể.

mpPose = mp.solutions.pose # Đối tượng của lớp Pose trong mediapipe để phát hiện các điểm đặc trưng trên cơ thể.
pose = mpPose.Pose() # Đối tượng Pose để sử dụng cho việc xử lý hình ảnh.
mpDraw = mp.solutions.drawing_utils # Đối tượng của lớp drawing_utils trong mediapipe để vẽ các điểm và đường nối trên hình ảnh.

model = tf.keras.models.load_model("model.h5") 

cap = cv2.VideoCapture(0) 

def make_landmark_timestep(results): #Chuyển đổi các điểm đặc trưng từ đối tượng results
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):#Vẽ các điểm đặc trưng và đường nối trên hình ảnh
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):#Vẽ nhãn hành động 
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):#Dự đoán nhãn hành động bằng cách sử dụng mô hình 
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)

    if results[0][0] > normal_threshold:
        label = "WAVING HAND"
    elif results[0][1] > normal_threshold:
        label = "FAN DANCE"
    elif results[0][3] > 0.6:
        label = "NORMAL"
    elif results[0][2] > 0.88:
        label = "SHOOK"
    else:
        label = "UNDEFINE "
    return label

i = 0
warmup_frames = 30

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            print("do dai c_lm : ", len(lm_list))
            print("Do dai steps : " ,n_time_steps)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()