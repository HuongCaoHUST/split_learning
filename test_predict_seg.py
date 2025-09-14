from ultralytics import YOLO
import cv2

# Load a segmentation model
model = YOLO("./merged_model.pt")

# Run prediction
results = model("https://ultralytics.com/images/bus.jpg")

# Hiển thị kết quả
for result in results:
    # Vẽ mask + bounding box
    im_mask = result.plot()  
    
    # Hiển thị bằng OpenCV
    cv2.imshow("YOLOv8 Segmentation", im_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
