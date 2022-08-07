from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image  


model = load_model('cnns.h5')                    #load weights  

face_classifier = cv2.CascadeClassifier('haar.xml')

cap = cv2.VideoCapture(0)

app = Flask(__name__)
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  

def gen_frames():                                       # generate frame by frame from cap
    while True:
        # Capture frame by frame
        success, frame = cap.read()
        if not success:
            break
        else:
            gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
            faces = face_classifier.detectMultiScale(gray)  
            
        
            for (x,y,w,h) in faces:
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)  
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
                # img_pixels = image.img_to_array(roi_gray)  
                # img_pixels = np.expand_dims(img_pixels, axis = 0)  
                # img_pixels /= 255  
        
                # predictions = model.predict(img_pixels)  
        
                # max_index = np.argmax(predictions[0])   #find max indexed array
        
                
                # predicted_emotion = emotions[max_index]  
                
                # cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = image.img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                    prediction = model.predict(roi)[0]
                    label=emotions[prediction.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            resized_img = cv2.resize(frame, (1000, 700))  
            
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    