import cv2
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor

# Load the trained model
model = torch.hub.load('client/yolov5', 'custom', path='client/yolov5/runs/train/exp25/weights/best.pt', source='local')

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define the transformation
transform = Compose([Resize((640, 640)), ToTensor()])

# Start the webcam
cap = cv2.VideoCapture(0)
cv2.waitKey()

while True:
    # Read the frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        break

    # Convert the frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the transformation and add an extra dimension
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Perform the object detection
    with torch.no_grad():
        detections = model(img_tensor)

    # For each detection
    for detection in detections.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw a rectangle around the object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()

