import cv2
import numpy as np

def draw_circle(img, center, radius, color, orientation):
    orientation_rad = np.deg2rad(orientation)
    gradient_circle = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    for y in range(center[1] - radius, center[1] + radius):
        for x in range(center[0] - radius, center[0] + radius):
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                if distance <= radius:
                    angle_to_center = np.arctan2(y - center[1], x - center[0])
                    angle_diff = np.abs(np.arctan2(np.sin(angle_to_center - orientation_rad), np.cos(angle_to_center - orientation_rad)))
                    opacity = int(np.clip(np.cos(angle_diff), 0, 1) * 255)
                    modulated_color = [int(c) for c in color] + [opacity]  # RGBA
                    gradient_circle[y, x] = modulated_color
    return gradient_circle

def visualize_sift(img):
    black_background = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, _ = sift.detectAndCompute(gray, None)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        orientation = kp.angle

        x_start = max(0, x - radius)
        y_start = max(0, y - radius)
        x_end = min(img.shape[1], x + radius)
        y_end = min(img.shape[0], y + radius)
        region = img[y_start:y_end, x_start:x_end]
        mean_color = region.mean(axis=(0, 1)).astype(int)  # Mean color (B, G, R)

        gradient_circle = draw_circle(
            black_background, (x, y), radius, mean_color, orientation
        )
        alpha = gradient_circle[:, :, 3] / 255.0
        for c in range(3):  # Blend RGB channels
            black_background[:, :, c] = (1.0 - alpha) * black_background[
                :, :, c
            ] + alpha * gradient_circle[:, :, c]
        black_background[:, :, 3] = np.maximum(
            black_background[:, :, 3], gradient_circle[:, :, 3]
        )

    blurred_image = cv2.GaussianBlur(black_background[:, :, :3], (15, 15), 0)
    blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
    return blurred_image_rgb, keypoints
