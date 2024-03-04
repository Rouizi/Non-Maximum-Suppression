import cv2
import numpy as np


image = cv2.imread("images/1.jpg")
# Take a copy of the original image to draw the bounding boxes after NMS
image_copy = image.copy()

# Dummy data: boxes in format (xmin, ymin, xmax, ymax), scores, and threshold
boxes = np.array([[245, 305, 575, 490],   # Box 1 coordinates
                  [235, 300, 485, 515],   # Box 2 coordinates, overlaps with Box 1
                  [305, 270, 540, 500],]) # Box 3 coordinates, overlaps with Box 1
confidence_scores = np.array([0.9, 0.8, 0.6]) 
threshold = 0.5

# Draw the bounding boxes on the image
for xmin, ymin, xmax, ymax in boxes:
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

cv2.imshow("Before NMS", image)
cv2.waitKey(0)


# Apply non-maximum suppression
indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidence_scores, score_threshold=0.7, nms_threshold=threshold)

# Filter out the boxes based on the NMS result
filtered_boxes = [boxes[i] for i in indices.flatten()]

# Draw the filtered boxes on the image
for xmin, ymin, xmax, ymax in filtered_boxes:
    cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

cv2.imshow("After NMS", image_copy)
cv2.waitKey(0)

print("Filtered Boxes:", filtered_boxes)

