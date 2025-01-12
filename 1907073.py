import numpy as np
import cv2
import pyautogui

background = None
frames_elapsed = 0
FRAME_HEIGHT = 600
FRAME_WIDTH = 600
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH

class HandData:
    def __init__(self):
        self.contour = None
        self.hull = None
        self.defects = None

def write_on_image(frame, finger_count):
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region_blur = cv2.GaussianBlur(region_gray, (5, 5), 0)
    return region, region_blur

def get_average(region_blur):
    global background
    if background is None:
        background = region_blur.copy().astype("float")
        return
    cv2.accumulateWeighted(region_blur, background, BG_WEIGHT)

def segment(region, region_blur):
    global hand
    diff = cv2.absdiff(background.astype(np.uint8), region_blur)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    thresholded_region = cv2.morphologyEx(thresholded_region, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresholded_region = cv2.morphologyEx(thresholded_region, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour

def get_skin_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

    lower_hsv = np.array([0, 48, 80], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    skin_mask = cv2.bitwise_or(skin_mask_ycrcb, skin_mask_hsv)

    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    skin_mask = cv2.medianBlur(skin_mask, 5)

    return skin_mask

def combine_masks(background_mask, skin_mask):
    combined_mask = cv2.bitwise_and(background_mask, skin_mask)
    return combined_mask

def calculate_finger_count(hand):
    finger_count = 0
    if hand.contour is not None and hand.defects is not None:
        for i in range(hand.defects.shape[0]):
            s, e, f, d = hand.defects[i, 0]
            start = tuple(hand.contour[s][0])
            end = tuple(hand.contour[e][0])
            far = tuple(hand.contour[f][0])
            
            # Convert points to NumPy arrays
            start = np.array(start)
            end = np.array(end)
            far = np.array(far)

            # Calculate the angle between the start, end, and far points
            a = np.linalg.norm(end - start)
            b = np.linalg.norm(far - start)
            c = np.linalg.norm(end - far)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

            # Distance between start and end points
            distance = np.linalg.norm(start - end)

            # Filter defects by angle and distance
            if angle <= np.pi / 2 and distance > 20:  # Adjust the distance threshold as needed
                finger_count += 1

        finger_count = min(finger_count, 5)  # Limit to 5 fingers

    return finger_count + 1  # Add 1 to account for the thumb or base of the hand

def main():
    global frames_elapsed, OBJ_THRESHOLD
    capture = cv2.VideoCapture(0)
    hand = HandData()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)

        region, region_blur = get_region(frame)
        cv2.imshow("Region", region)
        cv2.imshow("Region Blur", region_blur)

        finger_count = 0  # Initialize finger count to 0 at the start of each loop iteration

        if frames_elapsed < CALIBRATION_TIME:
            get_average(region_blur)
        else:
            hand_contour = segment(region, region_blur)
            if hand_contour is not None:
                skin_mask = get_skin_mask(region)
                cv2.imshow("Skin Mask", skin_mask)

                background_mask = np.zeros_like(skin_mask)
                cv2.drawContours(background_mask, [hand_contour], -1, 255, thickness=cv2.FILLED)
                cv2.imshow("Background Mask", background_mask)

                combined_mask = combine_masks(background_mask, skin_mask)
                cv2.imshow("Combined Mask", combined_mask)
                
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                cv2.imshow("Combined Mask after Morphology", combined_mask)
                
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    hand_contour = max(contours, key=cv2.contourArea)
                    
                    x, y, w, h = cv2.boundingRect(hand_contour)
                    aspect_ratio = w / float(h)
                    area = cv2.contourArea(hand_contour)
                    rect_area = w * h
                    extent = area / float(rect_area)

                    if 0.2 < aspect_ratio < 1.8 and 0.3 < extent < 0.9:
                        cv2.drawContours(region, [hand_contour], -1, (255, 255, 255))
                        cv2.imshow("Segmented Image", region)

                        hand.contour = hand_contour
                        hand.hull = cv2.convexHull(hand_contour, returnPoints=False)
                        hand.defects = cv2.convexityDefects(hand.contour, hand.hull)
                        finger_count = calculate_finger_count(hand)
                    else:
                        finger_count = 0  # No hand detected, set finger count to 0
                else:
                    finger_count = 0  # No hand detected, set finger count to 0
            else:
                finger_count = 0  # No hand detected, set finger count to 0

        write_on_image(frame, finger_count)
        if finger_count == 1:
            pyautogui.keyDown('right')
        else:
            pyautogui.keyUp('right')
        if finger_count == 2:
            pyautogui.keyDown('left')
        else:
            pyautogui.keyUp('left')
        if finger_count == 3:
            pyautogui.press('enter')
        
            
        
        cv2.imshow("Camera Input", frame)
        frames_elapsed += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            break
        elif key == ord('u'):
            OBJ_THRESHOLD += 1
            print(f"Increasing OBJ_THRESHOLD to {OBJ_THRESHOLD}")
        elif key == ord('d'):
            OBJ_THRESHOLD -= 1
            print(f"Decreasing OBJ_THRESHOLD to {OBJ_THRESHOLD}")

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
