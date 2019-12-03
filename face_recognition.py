#Face Recognition
#Importing Libraries
import cv2

#Loading cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #Cascade for the face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #Cascade for the eye

# Defining a function that will do the detection
def detect(gray,frame): #this function takes gray and original image as input
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #To locate one or several faces
    for (x,y,w,h) in faces:  #For each detected face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,20,147)) #Paint a rectangle around the face
        gray_im = gray[y:y+h,x:x+w] #get the region of interest in the black and white image
        colorful_im = frame[y:y+h,x:x+w] #get the region of interest in the original image
        eyes = eye_cascade.detectMultiScale(gray_im,1.3,5)
        for (x_e,y_e,w_e,h_e) in eyes:  #For each detected eye
            cv2.rectangle(colorful_im,(x_e,y_e),(x_e+w_e,y_e+h_e),(255,0,0),2) #Paint a rectangle around the afce
            
        return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)  #Turn web cam on
while True:
    _, frame = video_capture.read()  #Get the last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #some colour transformations
    canvas = detect(gray, frame)  #get the output of our detect function
    cv2.imshow('Video', canvas)  #display the outputs
    if cv2.waitKey(1) & 0xFF == ord('q'):   #PLEASE PRESS "q" TO QUIT FROM THE CAMERA
        break
video_capture.release()
cv2.destroyAllWindows()


