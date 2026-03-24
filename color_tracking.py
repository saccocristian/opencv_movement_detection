import cv2
import numpy as np

def main():
    print("Hello openCV")
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        lowerb = np.array([5,160,100])
        upperb = np.array([18,255,255])
        frame_final = cv2.inRange(frame_hsv,lowerb, upperb)

        contours, _ = cv2.findContours(frame_final,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2500:
                continue
            (x,y,w,h) = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            first_frame = None

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()