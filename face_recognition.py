import cv2 as cv 


haar_cascade=cv.CascadeClassifier(r'C:\Users\adwai\Python Adwaith\.dist\haar_face.xml')

people=["Cristiano ronaldo","Lionel messi","Mohanlal","Sachin tendulkar"]


face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\adwai\Python Adwaith\.dist\face_trained.yml')

#img=cv.imread(r'C:\Users\adwai\Pictures\Saved Pictures\gettyimages-90968086-612x612.jpg')
video=cv.VideoCapture(0)
while True:
    ret,frame=video.read()


    gray_img=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


#detect the face in the image
    faces_rect=haar_cascade.detectMultiScale(gray_img,1.1,9)


    for x,y,w,h in faces_rect:

        faces_roi=gray_img[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(faces_roi)

        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=3)
        cv.putText(frame,f'{people[label]} {confidence}%',(x,y),cv.FORMATTER_FMT_DEFAULT,1.0,(255,255,40),thickness=2)
        print(confidence)
        print(people[label])

    cv.imshow('detected face',frame)
        
    if cv.waitKey(20) & 0xff==ord('d'):

        break


video.release()

cv.destroyAllWindows()










