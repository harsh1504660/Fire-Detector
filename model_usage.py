# %%
import pygame as py
import keras.preprocessing
from keras.preprocessing import image
import PIL
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# %%
vedio = cv2.VideoCapture(0)

# %%
width = int(vedio.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vedio.get(cv2.CAP_PROP_FRAME_HEIGHT))

# %%
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
font = cv2.FONT_HERSHEY_SIMPLEX
from keras.preprocessing import image
#Load the saved model
model = tf.keras.models.load_model('FireDetector-v5.h5')
video = cv2.VideoCapture(0)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
        _, frame = video.read()
#Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
#Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = tf.keras.preprocessing.image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        #if prediction is 0, which means there is fire in the frame.
        if prediction == 0:
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),20)
            cv2.putText(frame,"FIRE DETECTED", (100,120), font, 2, (0,0,255))
            cv2.rectangle(frame,(80,50),(600,150),(0,0,255),2)
            py.init()
            py.mixer.init()
            sounda= py.mixer.Sound('Fire_alarm_sound.wav')
            sounda.play()
        else:
            cv2.putText(frame,"NOT FIRE", (180,120), font, 2, (0,255,0))
            cv2.rectangle(frame,(500,50),(150,150),(0,255,0),2)
            if py.mixer.get_busy()==True:
                py.mixer.stop()
            else:
                pass
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()

# %%
