import cv2
import numpy as np
from openvino_inference.detection import ModelDetection
from openvino_inference.face_landmark import FacialLandmark


if __name__ == '__main__':
    ## -----------------------------------------
    ## Config
    ## -----------------------------------------
    # video
    video_src = 0
    # config -> visualize
    color_fd = (0, 150, 250)
    # model
    pd_path = 'model/face-detection-adas-0001'
    ag_path = 'model/facial-landmarks-35-adas-0002'
    device = "CPU"
    cpu_extension = None
    # visualize
    view_detection = True
    ## -----------------------------------------
    ## Init
    ## -----------------------------------------
    # Model
    th_detection = 0.7
    detection = ModelDetection(model_name=pd_path, device=device, extensions=cpu_extension, threshold = th_detection)
    detection.load_model()
    facial_landmark_detection = FacialLandmark(model_name=ag_path, device=device, extensions=cpu_extension)
    facial_landmark_detection.load_model()        
    # Video
    cap = cv2.VideoCapture(video_src)
    ## -----------------------------------------
    ## Program Start
    ## -----------------------------------------    
    while cap.isOpened():
        ret, frame = cap.read()        
        ## Detection 
        boxes = []
        boxes, scores = detection.predict(frame)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            face = frame[ymin:ymax, xmin:xmax]
            ## Age Gender
            landmarks = facial_landmark_detection.predict(face)
            # Visualization Detection
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,255,0), 2)
            for point in landmarks:
                cv2.circle(frame, (point[0]+xmin, point[1]+ymin), 3, (255,255,255))
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
