from Functions import *

model = InitModel()
num = 0


if __name__ == "__main__":
    while num < 10:
        
        path1 = f"C:/Users/GRiM/Python Projects/Sudoku Project/Resources/digits/digit{num}.png"
        path2 = f"C:/Users/GRiM/Python Projects/Sudoku Project/Resources/DRAWN/digit{num}.png"
        
        try:
            img_set1 = cv2.imread(path1)[:,:,0]
            img_set1 = np.invert(np.array([img_set1]))
            prediction = model.predict(img_set1)
            print("Pic:", num, "Digit:", np.argmax(prediction), "~", np.amax(prediction)*100, "%")

            img_set2 = cv2.imread(path1)[:,:,0]
            img_set2 = np.invert(np.array([img_set2]))
            prediction = model.predict(img_set2)
            print("Pic:", num, "Drawing:", np.argmax(prediction), "~", np.amax(prediction)*100, "%")

            print('\n')

        except:
            print("error")
            
        finally:    
            num += 1

        


