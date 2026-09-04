import cv2

def preprocess_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Starter preprocessing. We will tune this using real ship/gauge images.
    image = cv2.resize(image, None, fx=1.0, fy=1.0)
    return image
