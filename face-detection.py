import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
    img_gray, 
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w, y+h),(255,0,0),2)

    cv2.imshow("Faces",img)
    key = cv2.waitKey(1)

    if(key == 13):
        break;

cv2.waitKey(0)
cv2.destroyAllWindows()



