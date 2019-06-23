import cv2
import os
import winsound
import pyautogui as py


#Classifiers

fbody_cascade=cv2.CascadeClassifier("haarcascade_fullbody.xml")
lbody_cascade=cv2.CascadeClassifier("haarcascade_lowerbody.xml")
ubody_cascade=cv2.CascadeClassifier("haarcascade_upperbody.xml")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 

#setting frequency and duration

frequency = 2500          # Set Frequency To 2500 Hertz
duration = 1000# Set Duration To 1000 ms == 1 second



cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*"XVID")
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

while True:
    ret, img=cap.read()
    #out.write(img) 

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

    fbody=fbody_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)    
    for (fx,fy,fw,fh) in fbody:
        cv2.rectangle(img,(fx,fy),(fx+fw,fy+fh),(0,255,0),4)

    lbody=lbody_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    for (lx,ly,lw,lh) in lbody:
        cv2.rectangle(img,(lx,ly),(lx+lw,ly+lh),(0,255,0),4)

    ubody=lbody_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    for (ux,uy,uw,uh) in ubody:
        cv2.rectangle(img,(ux,uy),(ux+uw,uy+uh),(0,255,0),4)

    faces = face_cascade.detectMultiScale(gray, scaleFactor =1.1, minNeighbors = 5)
    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        #start to capture an image
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]         
        eyes = eye_cascade.detectMultiScale(roi_gray)        
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,175,255),2)

    lf=len(fbody)
    ll=len(lbody)
    lu=len(ubody)
    lff=len(faces)

    if ((lf>0) or (ll>0) or (lu>0) or (lff>0)):
        out.write(img)
        py.screenshot("Defaulter.png")
        winsound.Beep(frequency, duration)
    
    cv2.imshow('img',img)
    

    k=cv2.waitKey(1) & 0xff 
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()  




    

    

    
 
        

    
