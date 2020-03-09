import cv2

face_cascade = '../Resources/face_classifier.xml'
classifier = cv2.CascadeClassifier(face_cascade)


def get_boxes(image):
    global classifier
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = classifier.detectMultiScale(grey, 1.1, 4)
    return rects


def main():
    global classifier
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # This flag stops annoying warning
    while True:
        ret, orig_img = cap.read()
        orig_img = cv2.resize(orig_img, (1024, 768))
        grey = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        rects = classifier.detectMultiScale(grey, 1.1, 4)

        for x, y, w, h in rects:
            cv2.rectangle(orig_img, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=2)
        cv2.imshow("Image", orig_img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            exit()


if __name__ == "__main__":
    main()
