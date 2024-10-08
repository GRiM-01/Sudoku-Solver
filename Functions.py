# Suppress TensorFlow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)


import cv2
import numpy as np
import keras
import time

def InitModel():
    model = keras.models.load_model("C:/Users/GRiM/Python Projects/Sudoku Project/DLmodels/CNN-DigitRecog-V5.h5") # Replace Path
    return model

def imgProcess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_threshold

# Reorder Corners for Warping
def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def largestContour(contours):
    largest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            if area > max_area and len(approx) == 4:
                largest = approx
                max_area = area
    return largest, max_area

def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            box = box[4:box.shape[0] - 4, 4:box.shape[1] - 4]
            box = cv2.resize(box, (28, 28))
            boxes.append(box)
    return boxes

def digitRecog(boxes, model):
    result = []
    images = []
    for img in boxes:
        img = np.invert(np.array([img]))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        images.append(img)

        prediction = model.predict(img)
        classIndex = np.argmax(prediction, axis=1)
        probability = np.amax(prediction)

        if probability >= 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)

    return result, images

def stackImgs(img_Array, scale):
    rows = len(img_Array)
    cols = len(img_Array[0])
    availabe_rows = isinstance(img_Array[0], list)
    width = img_Array[0][0].shape[1]
    height = img_Array[0][0].shape[0]
    if availabe_rows:
        for x in range(0, rows):
            for y in range(0, cols):
                img_Array[x][y] = cv2.resize(img_Array[x][y], (0, 0), None, scale, scale)
                if len(img_Array[x][y].shape) == 2:
                    img_Array[x][y] = cv2.cvtColor(img_Array[x][y], cv2.COLOR_GRAY2BGR)
        blank_img = np.zeros((height, width, 3), np.uint8)
        hori = [blank_img] * rows
        hori_con = [blank_img] * rows

        for x in range(0, rows):
            hori[x] = np.hstack(img_Array[x])
            hori_con[x] = np.concatenate(img_Array[x])
        verti = np.vstack(hori)
    else:
        for x in range(0, rows):
            img_Array[x] = cv2.resize(img_Array[x], (0, 0), None, scale, scale)
            if len(img_Array[x].shape) == 2:
                img_Array[x] = cv2.cvtColor(img_Array[x], cv2.COLOR_GRAY2BGR)
        hori = np.hstack(img_Array)
        hori_con = np.concatenate(img_Array)
        verti = hori
    return verti

def displayNumbers(img, numbers, color=(0, 255, 0), fontScale=1, thickness=2):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)

    for x in range(9):
        for y in range(9):
            index = (y * 9) + x
            if numbers[index] != 0:
                text = str(numbers[index])
                org = (x * secW + int(secW / 2) - 10, int((y + 0.72) * secH)) #Text placement and alignment
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)

    return img


selected_box = None
def mouseCallback(event, x, y, flags, param):
    global selected_box
    box_coords, digits = param
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(box_coords):
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                selected_box = i
                print(f"Box {i+1} selected for editing.")

def editDetected(img, digits, box_coords):
    global selected_box
    while True:
        detected_digits_img = img.copy()
        detected_digits_img = displayNumbers(detected_digits_img, digits, color=(50, 120, 255), fontScale=1.2, thickness=4)

        cv2.imshow("Original", img)
        cv2.imshow("Edit Detected Digits", detected_digits_img)

        cv2.setMouseCallback("Edit Detected Digits", mouseCallback, param=(box_coords, digits))

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to break
            cv2.destroyAllWindows()
            break
            
        elif selected_box is not None:
            if key in range(48, 58):  # Number keys 0-9
                digits[selected_box] = key - 48
                print(f"Digit in box {selected_box} changed to {digits[selected_box]}")
                selected_box = None
                
    return digits



# Sudkou Solver

def valid(grid, row, col, number):
    for x in range(9):
        if grid[row][x] == number:
            return False       

    for x in range(9):
        if grid[x][col] == number:
            return False
        
    corner_row = row - row % 3
    corner_col = col - col % 3

    for i in range(3): 
        for j in range(3): 
            if grid[corner_row + i][corner_col + j] == number:
                return False
                
    return True

def solveSudoku(grid, row, col):
    if col == 9:
        if row == 8:
            return True
        
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)
    
    for num in range(1, 10):
        if valid(grid, row, col, num):
            grid[row][col] = num
            if solveSudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False
