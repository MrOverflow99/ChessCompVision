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
        # Screenshot
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # sava lo screen
        full_screenshot_path = os.path.join(output_dir, "debug_screenshot.jpg")
        cv2.imwrite(full_screenshot_path, img_bgr)
        print(f"Full screenshot saved as {full_screenshot_path}")
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Salva img
        blurred_path = os.path.join(output_dir, "debug_blurred.jpg")
        thresh_path = os.path.join(output_dir, "debug_thresh.jpg")
        morph_path = os.path.join(output_dir, "debug_morph.jpg")
        cv2.imwrite(blurred_path, blurred)
        cv2.imwrite(thresh_path, thresh)
        cv2.imwrite(morph_path, morph)
        
        # Trova contorni
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # ... e disegnali per il debug!
        contour_img = img_bgr.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
        contours_path = os.path.join(output_dir, "debug_contours.jpg")
        cv2.imwrite(contours_path, contour_img)
        
        # trova il quadrato piu grande
        chessboard_contour = None
        max_area = 0
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    chessboard_contour = approx
        
        if chessboard_contour is not None:
            # Trova angoli
            corners = chessboard_contour.reshape(4, 2)
            

            sorted_box = sorted(corners, key=lambda p: p[0] + p[1])
            tl, tr, br, bl = sorted_box[0], sorted_box[1], sorted_box[2], sorted_box[3]
            
            
            min_x = max(0, int(min(tl[0], bl[0])))
            max_x = min(img_bgr.shape[1], int(max(tr[0], br[0])))
            min_y = max(0, int(min(tl[1], tr[1])))
            max_y = min(img_bgr.shape[0], int(max(bl[1], br[1])))
            
            
            margin = 10
            min_x = max(0, min_x - margin)
            max_x = min(img_bgr.shape[1], max_x + margin)
            min_y = max(0, min_y - margin)
            max_y = min(img_bgr.shape[0], max_y + margin)
            
            
            cropped = img_bgr[min_y:max_y, min_x:max_x]
            
            # Salva la foto
            cropped_path = os.path.join(output_dir, "chessboard_cropped.jpg")
            cv2.imwrite(cropped_path, cropped)
            print(f"Cropped chessboard saved as {cropped_path}")
            
            # Mostra la foto
            cv2.imshow('Cropped Chessboard', cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Chessboard not found.")

if __name__ == "__main__":
    main()