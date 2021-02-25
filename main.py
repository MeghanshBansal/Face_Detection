import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    smiles = tuple()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, z, w) in faces:
        face = gray[y:y + z, x:x + w]
        smiles = smile_cascade.detectMultiScale(face, 1.8, 20)
    cv2.imshow('img', img)

    if len(smiles)>=1:
        cv2.imwrite('smile_snap.jpg', img)
        break

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
