import cv2

face_cascade = 'face_classifier.xml'

classifier = cv2.CascadeClassifier(face_cascade)

cap = cv2.VideoCapture(0)

while True:
    ret, orig_img = cap.read()
    grey = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    rects =  classifier.detectMultiScale(grey, 1.1, 4)
    for x,y,w,h in rects:
        cv2.rectangle(orig_img, (x,y), (x+w, y+h), color=(255,0,0), thickness=2)
    
    cv2.imshow("Image", orig_img)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        exit()