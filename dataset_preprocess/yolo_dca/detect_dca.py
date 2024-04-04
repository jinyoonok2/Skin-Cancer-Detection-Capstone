import cv2
def detect_dark_corners(image, corner_size=10, darkness_threshold=10, dark_pixel_ratio=0.3):
    """
    Classify if an image has dark corners and visualize the corners if dark corners are detected.

    Parameters:
    - image: The image to classify.
    - image_name: The name of the image file (used for visualization saving).
    - corner_size: The size of the corner to analyze.
    - darkness_threshold: The threshold for considering a pixel as dark.
    - dark_pixel_ratio: The ratio of dark pixels required to classify a corner as dark.

    Returns:
    - True if the image has dark corners, False otherwise.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = {
        'top_left': gray[:corner_size, :corner_size],
        'top_right': gray[:corner_size, -corner_size:],
        'bottom_left': gray[-corner_size:, :corner_size],
        'bottom_right': gray[-corner_size:, -corner_size:]
    }

    dark_corners_detected = 0
    for _, corner in corners.items():
        dark_pixels = np.sum(corner < darkness_threshold)
        if dark_pixels / (corner_size * corner_size) > dark_pixel_ratio:
            dark_corners_detected += 1

    return dark_corners_detected > 0
