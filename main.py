from Functions import *

start_time = time.time()

img_height = 540
img_width = 540
model = InitModel()

if __name__ == "__main__":
    print("Processing...")

    img_name = '4.jpg' # Replace image name

    img = cv2.imread(f"C:/Users/GRiM/Python Projects/Sudoku Project/Images/{img_name}") # Replace path
    img = cv2.resize(img, (img_width, img_height))
    blank_img = np.zeros((img_height, img_width, 3), np.uint8)

    # Img Processing
    img_threshold = imgProcess(img)
    img_contours = img.copy()
    img_corners = img.copy()
    contours, hierachy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # Find largest contour
    largest, max_area = largestContour(contours)
    if largest.size != 0:
        largest = reorder(largest)
        cv2.drawContours(img_corners, largest, -1, (0, 0, 255), 25)
        points1 = np.float32(largest)
        points2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
        corners = cv2.getPerspectiveTransform(points1, points2)

        img_warped_board = cv2.warpPerspective(img, corners, (img_width, img_height))
        img_detected_digits = img_warped_board.copy()
        result_img = img_warped_board.copy()
        img_warped_board = cv2.cvtColor(img_warped_board, cv2.COLOR_BGR2GRAY)
        solved_sudoku = blank_img.copy()

    # Split sudoku into 81 boxes
    boxes = splitBoxes(img_warped_board)

    # ML digit recognition
    result, images = digitRecog(boxes, model)
    unsolved_sudoku_board = np.array(result).reshape(9, 9)
    result = np.asarray(result)

    # Box co-ordinates
    box_coords = []
    box_size = img_width // 9
    for i in range(9):
        for j in range(9):
            x1, y1 = j * box_size, i * box_size
            x2, y2 = x1 + box_size, y1 + box_size
            box_coords.append((x1, y1, x2, y2))

    print("\nModel digit detection and classification completed. Please validate the results.\n")

    edited_digits = editDetected(img_detected_digits, result.copy(), box_coords)
    edited_digits2 = edited_digits.copy()

    unsolved_sudoku_board = np.array(edited_digits).reshape(9, 9)

    lol_result = [edited_digits[i:i + 9] for i in range(0, 81, 9)]
    missing_digits_placeholder = np.where(edited_digits > 0, 0, 1)
    missing_digits_mask = np.array(missing_digits_placeholder).reshape(9, 9)

    solved_puzzle = []
    if solveSudoku(lol_result, 0, 0):
        for i in range(9):
            for j in range(9):
                solved_puzzle.append(lol_result[i][j])

    # CHeck if solvable
    try: 

        arr_solved = np.array(solved_puzzle).reshape(9, 9)
        solved_puzzle_array = arr_solved * missing_digits_mask
        solved_digits = solved_puzzle_array.flatten().tolist()

        img_detected_digits = displayNumbers(img_detected_digits, edited_digits2, color=(50, 120, 255), fontScale=1.2, thickness=5)
        solved_sudoku = displayNumbers(solved_sudoku, solved_digits, color=(255, 0, 0), fontScale=1.2, thickness=3)

        result_img = cv2.addWeighted(result_img, 1, solved_sudoku, 0.5, 1)
        result_img = displayNumbers(result_img, solved_digits, color=(255, 150, 50), fontScale=1.2, thickness=2)

        img_Array = ([img, img_threshold, img_contours, img_corners],
                    [img_warped_board, img_detected_digits, solved_sudoku, result_img])

        img_Stack = stackImgs(img_Array, 1)
        cv2.imshow("PICS - Processing, Identification, Classification, Solution", img_Stack)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\nSolved!, Runtime: {execution_time:.2f} seconds")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError:
        
        img_detected_digits = displayNumbers(img_detected_digits, edited_digits, color=(50, 120, 255), fontScale=1.2, thickness=5)

        img_Array = ([img, img_threshold, img_contours, img_corners],
                    [img_warped_board, img_detected_digits, blank_img, blank_img])

        img_Stack = stackImgs(img_Array, 1)
        cv2.imshow("PICS - Processing, Identification, Classification, Solution", img_Stack)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\nUnsolvable!, Runtime: {execution_time:.2f} seconds")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
