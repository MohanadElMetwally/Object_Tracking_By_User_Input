# Object_Tracking_With_User_Input
 
This project demonstrates a simple object tracking system that uses YOLOv8 Nano along with user input to select the object of interest. The user selects a region (bounding box) on the first frame of a video, and then YOLOv8 tracks that object throughout the video.

## Features

- **User-Driven Initialization:**  
  The system allows you to select the target object on the first frame via an interactive ROI selection.
  
- **YOLOv8 Nano Tracking:**  
  Uses YOLOv8 Nano for fast object detection and tracking.
  
- **Real-Time Visualization:**  
  The video displays the tracked object with its corresponding ID.

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/MohanadElMetwally/Object_Tracking_With_User_Input.git
   ```
   
2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
