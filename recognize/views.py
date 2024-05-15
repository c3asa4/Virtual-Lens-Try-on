from django.shortcuts import render,redirect
import cv2
import numpy as np
import dlib
from glob import glob
import tensorflow as tf
from smartglass.settings import KERAS_FILE,LANDMARK_FILE,XML_FILE
import os

def home(request):
    return render(request,'index.html')

def explore(request):
    return render(request,'explore.html')

def signup(request):
    return render(request,'profile.html')

def try_on(request):
    
    face_shape = get_matching_glasses()
    images = []
    # outcome_folder = os.listdir('/staticfiles/outcome/oval')
    # print(outcome_folder)
    outcome_image = {'Oval': os.listdir('staticfiles/outcome/Oval'),
                     'Round': os.listdir('staticfiles/outcome/Round'),
                     'Oblong': os.listdir('staticfiles/outcome/Oblong'),
                     'Heart': os.listdir('staticfiles/outcome/Heart'),
                     'Square': os.listdir('staticfiles/outcome/Square')}
    final_glass = outcome_image.get(face_shape)
    model_name = [os.path.splitext(image_name)[0] for image_name in final_glass]
    print(model_name)
    params = {'face_shape':face_shape,'outcome_img':model_name}
    
    return render(request,'wear.html',params)
    # return render(request,'wear.html')

# Function to preprocess image for model input
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to match model input shape
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img



# Function to get face shape label
def get_face_shape(class_index):
    
    if class_index == 0:
        return 'Heart'
    elif class_index == 1:
        return 'Oblong'
    elif class_index == 2:
        return 'Oval'
    elif class_index == 3:
        return 'Round'
    elif class_index == 4:
        return 'Square'
    else:
        return 'Unknown'




def get_matching_glasses():
    # Open camera
    cap = cv2.VideoCapture(0)


        # Add text to the frame
        

    max_probability = 0
    max_prob_face_shape = ''
    # Load pre-trained face shape classification model
    model = tf.keras.models.load_model(KERAS_FILE)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    LANDMARK_FILE = '/shape_predictor_68_face_landmarks.dat'
    # Load facial landmark predictor
    app_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the .dat file inside the app directory
    model_path = os.path.join(app_dir, 'shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(model_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break
        text = "Press Q To Get Recommmendations"
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop over each face
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Detect facial landmarks
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # Extract face region
            face_region = frame[y:y+h, x:x+w]

            # Preprocess face image for model input
            preprocessed_face = preprocess_image(face_region)

            # Predict face shape using the pre-trained model
            predictions = model.predict(preprocessed_face)
            class_index = np.argmax(predictions)
            probability = np.max(predictions)
            face_shape = get_face_shape(class_index)

            # Display face shape label and probability
            cv2.putText(frame, f"{face_shape} ({probability:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Update maximum probability face shape
            if probability > max_probability:
                max_probability = probability
                max_prob_face_shape = face_shape

        # Display the resulting frame
        cv2.imshow('Face Shape Classification', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Face Shape with Highest Probability:", max_prob_face_shape)
    # outcome_image = {'oval':'oval.jpg','avaitor':'avaitor.jpg','circular':'circular.jpg','rectangular':'ractangular.jpg'}

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    return(max_prob_face_shape)






