import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.ops import box_iou

# Load YOLOv8 Nano
model = YOLO("yolov8n.pt")

video_path = "mall_security_camera.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Unable to open video.")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Unable to read the first frame.")
    exit()

# Let the user select an object on the first frame.
roi = cv2.selectROI("Select Object (Press ENTER/SPACE to confirm)", first_frame)
cv2.destroyWindow("Select Object (Press ENTER/SPACE to confirm)")

# Convert the ROI from (x, y, w, h) to (x1, y1, x2, y2)
x, y, w, h = roi
user_box = (x, y, x + w, y + h)

cap.release()

# Run YOLOv8 tracking on the video.
results = model.track(source=video_path, stream=True)

# Initialize target_track_id = None 
# This variable will store the id corresponding to the object selected by the user.
target_track_id = None

# Process each frame result from YOLOv8 Nano.
for result in results:
    # Copy the current frame to avoid drawing on any images in result.
    frame = result.orig_img.copy()

    # Extract boxes tensor from result
    boxes = result.boxes.xyxy.numpy() if result.boxes is not None else np.array([])

    # Extract ids tensor from result
    track_ids = result.boxes.id.numpy() if result.boxes is not None and result.boxes.id is not None else None

    # If no boxes are detected, show the frame and continue.
    if boxes.size == 0 or track_ids is None:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        continue

    # On the first frame where we have detections, match the user-selected box.
    if target_track_id is None:
        user_box_tensor = torch.tensor([list(user_box)])
        boxes_tensor = torch.tensor(boxes, dtype=torch.float)
        iou_matrix = box_iou(user_box_tensor, boxes_tensor)
        best_iou, best_idx = torch.max(iou_matrix, dim=1)
        best_iou = best_iou.item()
        best_idx = best_idx.item()

        if best_iou > 0.1:
            target_track_id = track_ids[best_idx]
            print(f"Target track id found: {target_track_id}")
    
    # Draw only the detection that matches the target_track_id
    for i, box in enumerate(boxes):
        if target_track_id is not None and track_ids[i] == target_track_id:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(target_track_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display all detections (Optional)
    # for box in boxes:
    #     x1, y1, x2, y2 = map(int, box)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
