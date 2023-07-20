import cv2
import os
import  sys
# Get the current script's directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory path (project_folder in this case)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to Python's sys.path
sys.path.append(parent_dir)
from camera.simoutaneous_sender import send_to_server




if __name__ == '__main__':
    #Nano,GOTURN, MIL, DaSiamRPN
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Unable to access the camera.")
       
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    ret, frame = video.read()
   
    # print(frame)
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # 
    
    cv2.imwrite('.../picture/first.jpg',frame)
    # Initialize video writer to save the results
    # output = cv2.VideoWriter(f'{tracker_type}.avi',
    #                          cv2.VideoWriter_fourcc(*'XVID'), 60.0,
    #                          (frame_width // 2, frame_height // 2), True)
    if not ret:
        print('cannot read the video')
    # Select the bounding box in the first frame
    video.release()
    send_to_server(type='init',img =frame)
    while True:
        pass


