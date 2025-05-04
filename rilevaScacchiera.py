import os
import numpy as np
import cv2
from mss import mss

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    output_dir = "immRilevate"
    ensure_directory_exists(output_dir)

    with mss() as sct:
        # Capture entire screen
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Save the full screenshot
        full_screenshot_path = os.path.join(output_dir, "debug_screenshot.jpg")
        cv2.imwrite(full_screenshot_path, img_bgr)
        print(f"Full screenshot saved as {full_screenshot_path}")

        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Save preprocessed images for debugging
        blurred_path = os.path.join(output_dir, "debug_blurred.jpg")
        thresh_path = os.path.join(output_dir, "debug_thresh.jpg")
        morph_path = os.path.join(output_dir, "debug_morph.jpg")
        cv2.imwrite(blurred_path, blurred)
        cv2.imwrite(thresh_path, thresh)
        cv2.imwrite(morph_path, morph)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours for debugging
        contour_img = img_bgr.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
        contours_path = os.path.join(output_dir, "debug_contours.jpg")
        cv2.imwrite(contours_path, contour_img)

        # Filter contours to find the largest quadrilateral
        chessboard_contour = None
        max_area = 0

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour is a quadrilateral
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    chessboard_contour = approx

        if chessboard_contour is not None:
            # Get corner points
            tl, tr, br, bl = chessboard_contour.reshape(4, 2)

            # Sort the points to get top-left, top-right, bottom-right, bottom-left
            sorted_box = sorted(chessboard_contour.reshape(4, 2), key=lambda p: p[0] + p[1])
            tl, tr, br, bl = sorted_box[0], sorted_box[1], sorted_box[2], sorted_box[3]

            # Calculate bounding box
            min_x = max(0, int(min(tl[0], bl[0])))
            max_x = min(img_bgr.shape[1], int(max(tr[0], br[0])))
            min_y = max(0, int(min(tl[1], tr[1])))
            max_y = min(img_bgr.shape[0], int(max(bl[1], br[1])))

            # Crop the chessboard
            cropped = img_bgr[min_y:max_y, min_x:max_x]

            # Save and display the result
            cropped_path = os.path.join(output_dir, "chessboard_cropped.jpg")
            cv2.imwrite(cropped_path, cropped)
            print(f"Cropped chessboard saved as {cropped_path}")
            
            # Optional: Show the cropped image
            cv2.imshow('Cropped Chessboard', cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Chessboard not found.")

if __name__ == "__main__":
    main()