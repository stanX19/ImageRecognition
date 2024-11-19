import cv2
import numpy as np
import imshow_resized

def detect_and_track_green(image, display_mask=False):
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if display_mask:
        imshow_resized.cv2_imshow_resized("Mask", mask)
    return image


def process_webcam():
    """
    Processes a live webcam feed to detect and track green objects.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        output_frame = detect_and_track_green(frame, display_mask=True)
        imshow_resized.cv2_imshow_resized("Green Ball Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path):
    """
    Processes a single image to detect and track green objects.

    Parameters:
    - image_path: Path to the input image.
    """
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}.")
        return

    # Detect and track green objects
    output_image = detect_and_track_green(image, display_mask=True)

    # Display the output image
    imshow_resized.cv2_imshow_resized("image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Uncomment one of the following to test:

    # Option 1: Process live webcam
    process_webcam()

    # Option 2: Process a single image
    # Provide the image path here
    process_image("../data/WhatsApp Image 2024-09-26 at 13.04.18_6d41ab91.jpg")
