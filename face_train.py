import cv2 as cv 
import os
import numpy as np 

people=["Cristiano ronaldo","Lionel messi","Mohanlal","Sachin tendulkar"]

DIR=r'C:\Users\adwai\Python Adwaith\opencv\imagesforcvision'

features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        for img in os.listdir(path):
            image_path=os.path.join(path,img)
            image_array=cv.imread(image_path)
            gray_img=cv.cvtColor(image_array,cv.COLOR_BGR2GRAY)

            haar_cascade=cv.CascadeClassifier(r'C:\Users\adwai\Python Adwaith\.dist\haar_face.xml')

            faces_rect=haar_cascade.detectMultiScale(gray_img,1.1,5)

            for x,y,w,h in faces_rect:
                faces_roi=gray_img[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
    

create_train()
print("training is done")
features=np.array(features,dtype='object')
labels=np.array(labels)
print(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#train the recognizer on the feature list and label list

face_recognizer.train(features,labels) 

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)










