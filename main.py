from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

app = Flask(__name__)

# Load your training data
training_data = pd.read_csv("C://Users//atchu//Downloads//Plant-Leaf-Disease-Prediction-main//Dataset//project.csv")

# Split data into features (X) and target variable (y)
X = training_data[['Temperature', 'Humidity', 'Rainfall', 'Ph', 'N', 'P', 'K']]
y = training_data['label']  # Replace 'Crop Name' with the actual column name in your dataset
X.fillna(X.mean(), inplace=True)
is_infinite = np.isinf(X).any()
print(is_infinite)
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Train your machine learning model
model = DecisionTreeClassifier()
model.fit(X, y)

#importing pickle files
model1 = pickle.load(open('Models/classifier.pkl','rb'))
ferti = pickle.load(open('Models/fertilizer.pkl','rb'))

filepath = 'C://Users//atchu//Downloads//Plant-Leaf-Disease-Prediction-main//Models//model.h5'
model2 = load_model(filepath)
print(model2)

print("Model Loaded Successfully")

def is_leaf_image(image_path, green_threshold=100):
    try:
        # img = Image.open(image_path)
        # img = img.convert('RGB')
        # img_array = np.array(img)

        # # Calculate the average green color intensity
        # green_intensity = np.mean(img_array[:, :, 1])  # Green channel is at index 1

        # # You can adjust the green_threshold based on your dataset
        # if green_intensity >= green_threshold:
        #     return True
        # else:
        #     return False
        
        
        # img = cv2.imread(image_path)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if len(contours) > 0:
        #     areas = [cv2.contourArea(c) for c in contours]
        #     max_index = np.argmax(areas)
        #     max_contour = contours[max_index]
        #     x, y, w, h = cv2.boundingRect(max_contour)

        #     # Checking if the detected contour area is reasonably within a leaf size
        #     leaf_area = img.shape[0] * img.shape[1] * 0.25  # Considering a minimum of 5% leaf area
        #     if cv2.contourArea(max_contour) > leaf_area:
        #         return True
        # return False

        # img = image.load_img(image_path, target_size=(224, 224))
        # img_array = image.img_to_array(img)
        # img_array = np.expand_dims(img_array, axis=0)
        # img_array = preprocess_input(img_array)

        # features = base_model.predict(img_array)
        # prediction = model2.predict(features)
        # return prediction[0] == 1

        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_green = np.array([25, 52, 72])  # Define the lower and upper bounds for green color in HSV
        upper_green = np.array([102, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)  # Create a mask for green color

        # Apply some morphological operations to eliminate noise and get a clearer mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Estimate the total area of all identified contours
            total_area = sum(cv2.contourArea(c) for c in contours)
            image_area = img.shape[0] * img.shape[1]  # Total area of the image

            # Consider an image as a 'leaf image' if the total contour area is more than 10% of the image area
            if total_area > 0.1 * image_area:
                return True

        return False

    except Exception as e:
        print("Error:", str(e))
        return False  # Handle any exceptions gracefully
def pred_tomato_dieas(tomato_plant):
    if not is_leaf_image(tomato_plant):
        pred=-1
    else:
        test_image = load_img(tomato_plant, target_size = (128, 128)) # load image 
        print("@@ Got Image for prediction")
    
        test_image = img_to_array(test_image)/255 # convert image to np array and normalize
        test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
    
        result = model2.predict(test_image) # predict diseased palnt or not
        print('@@ Raw result = ', result)
        pred = np.argmax(result, axis=1)
    print(pred)
    if pred==0:
        return "Tomato - Bacteria Spot Disease", 'Tomato-Bacteria Spot.html'
       
    elif pred==1:
        return "Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'
        
    elif pred==2:
        return "Tomato - Healthy and Fresh", 'Tomato-Healthy.html'
        
    elif pred==3:
        return "Tomato - Late Blight Disease", 'Tomato - Late_blight.html'
       
    elif pred==4:
        return "Tomato - Leaf Mold Disease", 'Tomato - Leaf_Mold.html'
        
    elif pred==5:
        return "Tomato - Septoria Leaf Spot Disease", 'Tomato - Septoria_leaf_spot.html'
        
    elif pred==6:
        return "Tomato - Target Spot Disease", 'Tomato - Target_Spot.html'
        
    elif pred==7:
        return "Tomato - Tomoato Yellow Leaf Curl Virus Disease", 'Tomato - Tomato_Yellow_Leaf_Curl_Virus.html'
    elif pred==8:
        return "Tomato - Tomato Mosaic Virus Disease", 'Tomato - Tomato_mosaic_virus.html'
        
    elif pred==9:
        return "Tomato - Two Spotted Spider Mite Disease", 'Tomato - Two-spotted_spider_mite.html'
    elif pred==-1:
        return "Error", 'error.html'

@app.route('/')
def welcome():
    return render_template('main.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('index1.html')

@app.route('/fertilizer/predict1',methods=['POST'])
def predict1():
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    input = [[int(temp),int(humi),int(mois),int(soil),int(crop),int(nitro),int(pota),int(phosp)]]

    res = ferti.classes_[model1.predict(input)]

    return render_template('index1.html',x = ('Predicted Fertilizer is {}'.format(res)))

@app.route("/leaf-disease")
def leaf_disease():
    return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/leaf-disease/predict2",methods=['POST'])
def predict2():
    if request.method == 'POST':
        file = request.files['image'] # fetch input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('C:/Users/atchu/Downloads/Plant-Leaf-Disease-Prediction-main/static/upload/', filename)
        file.save(file_path)
        
        print("@@ Predicting class......")
        pred, output_page = pred_tomato_dieas(file_path)     
        return render_template(output_page, pred_output = pred, user_image = file_path)

@app.route('/crop')
def crop():
    return render_template("recommendation.html")

@app.route('/crop/process_form', methods=['POST'])
def predict3():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Rainfall = float(request.form['Rainfall'])
        Ph = float(request.form['Ph'])
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])

        # Create a DataFrame from the input data
        input_data = pd.DataFrame({
            'Temperature': [Temperature],
            'Humidity': [Humidity],
            'Rainfall': [Rainfall],
            'Ph': [Ph],
            'N': [N],
            'P': [P],
            'K': [K]
        })

        # Perform predictions using the trained model
        new_predictions = model.predict(input_data)

        # You can use 'new_predictions' to display the results in the HTML template
        return render_template('recommendation.html',x=('Predicted Crop is {}'.format(new_predictions)))
    else:
        return "Invalid request."

if __name__ == "__main__":
    app.run(debug=True)