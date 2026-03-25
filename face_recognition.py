import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'Abhinav']  # yahan naam add kar sakte ho

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 100:
            name = names[id]
        else:
            name = "Unknown"

        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('camera', img)

    if cv2.waitKey(10) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()