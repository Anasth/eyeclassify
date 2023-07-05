from flask import Flask, request, jsonify
from tensorflow import keras

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from flask_cors import CORS
import cv2
import numpy as np


app = Flask(__name__)
CORS(app)

cataract_classification_model = keras.models.load_model("D:/Ocular/Catract/OcularCatract/my_model/ResNet50-Cataract_classification-98.98.h5")
glaucoma_classification_model = keras.models.load_model("D:/Ocular/Catract/OcularCatract/my_model/VGG19-Glaucoma_classification-98.91.h5")
retinopathy_classification_model = keras.models.load_model("D:/Ocular/Catract/OcularCatract/my_model/inception-DR_classification-95.00.h5")
amd_classification_model = keras.models.load_model("D:/Ocular/Catract/OcularCatract/my_model/VGG19-AMD_classification-84.21.h5")
# hypertension_classification_model = keras.models.load_model("D:/Ocular/Catract/OcularCatract/my_model/DenseNet201-Hypertension_classification-92.00.h5")


# TODO Edit just the file name after editing the file
# hypertension_classification_model = keras.models.load_model("D:/Ocular/Catract/OcularCatract/my_model/DenseNet201-Hypertension_classification-92.00.h5")


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): 
            return img 
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img



@app.route('/cataract', methods=["POST"])
def Cataract():

    image_size=224         

    def predict_class(image):


        # image = cv2.imread(image_path)
        image= crop_image_from_gray(image)
        image = cv2.resize(image,(image_size,image_size))
        
        image=np.array(image)
        image=image.reshape(-1,image_size,image_size,3)

        y = cataract_classification_model.predict([image])
        return y
    

    if 'imagefile' not in request.files:
        
        return "No image file found in the request." 
    

    else:
        print ("Image file received and processed successfully.")
    
    imagefile = request.files['imagefile']


    image_bytes = imagefile.read()

    image_np = np.frombuffer(image_bytes, np.uint8)
    
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if image is None:
        return 'Failed to load image', 400
    
    # Resize the image using cv2.resize()
    # resized_image = cv2.resize(image, (50, 50))
    # normalized_image = resized_image / 255.0
    # input_image = np.expand_dims(normalized_image, axis=0)
    # assert input_image.shape == (1, 255, 255, 3)

    # y = cataract_classification_model.predict([input_image])

    prediction = predict_class(image)  


    print(prediction[0][0])
    result = ''
    if prediction[0][0] > 0.45:
        print("Cataract")
        result = 'Cataract'
    else:
        print("Normal")
        result = 'Normal'

    # to fix the prediction value when it is less than (Normal) ? 
    if prediction[0][0] < 0.45:
        prediction[0][0] = 1 - prediction[0][0]
    
    prediction[0][0] = prediction[0][0] * 100
    prediction = prediction[0][0].tolist()  # Convert numpy array to a Python list
    prediction = round(prediction) 
    # Clean up the image file if needed
    print(prediction)
    print("***************************************************************************")
    
    return jsonify({'prediction': prediction, 'result': result})




# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////




@app.route('/glaucoma', methods=["POST"])
def Glaucoma():

    image_size=224         
    
    def predict_class(img):


        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        #configure CLAHE
        clahe = cv2.createCLAHE(clipLimit=10,tileGridSize=(8,8))

        #0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
        img[:,:,0] = clahe.apply(img[:,:,0])

        img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

        img= crop_image_from_gray(img)
        img = cv2.resize(img,(image_size,image_size))
        
        img=np.array(img)
        img=img.reshape(-1,image_size,image_size,3)
        img = img / 255.0
        y = glaucoma_classification_model.predict([img])
        return y

    if 'imagefile' not in request.files:
        
        return "No image file found in the request." 
    

    else:
        print ("Image file received and processed successfully.")
    
    imagefile = request.files['imagefile']


    image_bytes = imagefile.read()

    image_np = np.frombuffer(image_bytes, np.uint8)
    
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if image is None:
        return 'Failed to load image', 400
    

    prediction = predict_class(image)  


    print(prediction)
    result = ''
    if prediction[0][0] > 0.5:
        print("Glaucoma")
        result = 'Glaucoma'
    else:
        print("Normal")
        result = 'Normal'

    # to fix the prediction value when it is less than (Normal) ? 
    if prediction[0][0] < 0.5:
        prediction[0][0] = 1 - prediction[0][0]
    
    prediction[0][0] = prediction[0][0] * 100
    prediction = prediction[0][0].tolist()  # Convert numpy array to a Python list
    prediction = round(prediction) 
    # Clean up the image file if needed
    print(prediction)
    print("***************************************************************************")
    
    return jsonify({'prediction': prediction, 'result': result})




# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////



@app.route('/dr', methods=["POST"])
def DR():

    image_size=224         
    
    def predict_class(imge):
        
        # Blur the image
        blurred = cv2.blur(imge, ksize=(15, 15))

        # Take the difference with the original image
        # Weight with a factor of 4x to increase contrast
        dst = cv2.addWeighted(imge, 4, blurred, -4, 128)
        lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        gridsize = 5
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr2 =crop_image_from_gray(lab)
        bgr2 = cv2.cvtColor(bgr2, cv2.COLOR_LAB2BGR)
        bgr2 = cv2.resize(bgr2,(image_size,image_size))
        image =np.array(bgr2)/255          
        image=image.reshape(-1,image_size,image_size,3) 

        y = retinopathy_classification_model.predict([image])
        return y

    if 'imagefile' not in request.files:
        
        return "No image file found in the request." 
    

    else:
        print ("Image file received and processed successfully.")
    
    imagefile = request.files['imagefile']


    image_bytes = imagefile.read()

    image_np = np.frombuffer(image_bytes, np.uint8)
    
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if image is None:
        return 'Failed to load image', 400
    

    prediction = predict_class(image)  


    print(prediction)
    result = ''
    if prediction[0][0] > 0.5:
        print("DR")
        result = 'DR'
    else:
        print("Normal")
        result = 'Normal'

    # to fix the prediction value when it is less than (Normal) ? 
    if prediction[0][0] < 0.5:
        prediction[0][0] = 1 - prediction[0][0]
    
    prediction[0][0] = prediction[0][0] * 100
    prediction = prediction[0][0].tolist()  # Convert numpy array to a Python list
    prediction = round(prediction) 
    # Clean up the image file if needed
    print(prediction)
    print("***************************************************************************")
    
    return jsonify({'prediction': prediction, 'result': result})




# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////



@app.route('/amd', methods=["POST"])
def AMD():

    image_size=224         
    
    def predict_class(image):

        image= crop_image_from_gray(image)
        image = cv2.resize(image,(image_size,image_size))
        
        
        # Create a kernel for the unsharp mask filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # Apply the unsharp mask filter
        unsharp = cv2.filter2D(image, -1, kernel)
        unsharp= crop_image_from_gray(unsharp)
        unsharp = cv2.resize(unsharp,(image_size,image_size))
        image=unsharp.reshape(-1,image_size,image_size,3) 

        y = amd_classification_model.predict([image])
        return y

    if 'imagefile' not in request.files:
        
        return "No image file found in the request." 
    

    else:
        print ("Image file received and processed successfully.")
    
    imagefile = request.files['imagefile']


    image_bytes = imagefile.read()

    image_np = np.frombuffer(image_bytes, np.uint8)
    
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Check if the image was loaded successfully
    if image is None:
        return 'Failed to load image', 400
    

    prediction = predict_class(image)  


    print(prediction)
    result = ''
    if prediction[0][0] > 0.5:
        print("AMD")
        result = 'AMD'
    else:
        print("Normal")
        result = 'Normal'

    # to fix the prediction value when it is less than (Normal) ? 
    if prediction[0][0] < 0.5:
        prediction[0][0] = 1 - prediction[0][0]
    
    prediction[0][0] = prediction[0][0] * 100
    prediction = prediction[0][0].tolist()  # Convert numpy array to a Python list
    prediction = round(prediction) 
    # Clean up the image file if needed
    print(prediction)
    print("***************************************************************************")
    
    return jsonify({'prediction': prediction, 'result': result})





# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////



# @app.route('/hypertension', methods=["POST"])
# def Hypertension():

#     image_size=224         
    
#     def predict_class(image):
#         bits = tf.io.read_file(image)
#         image = tf.image.decode_jpeg(bits, channels=3) 
#         image = tf.image.resize(image, image_size)
#         image = tf.cast(image, tf.float32)
#         image = tf.image.per_image_standardization(image)
#         image= image.reshape(-1,image_size,image_size,3) 

#         y = hypertension_classification_model.predict([image])
#         return y

#     if 'imagefile' not in request.files:
        
#         return "No image file found in the request." 
    

#     else:
#         print ("Image file received and processed successfully.")
    
#     imagefile = request.files['imagefile']


#     image_bytes = imagefile.read()

#     image_np = np.frombuffer(image_bytes, np.uint8)
    
#     image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

#     # Check if the image was loaded successfully
#     if image is None:
#         return 'Failed to load image', 400
    

#     prediction = predict_class(image)  


#     print(prediction)
#     result = ''
#     if prediction[0][0] > 0.5:
#         print("Hypertension")
#         result = 'Hypertension'
#     else:
#         print("Normal")
#         result = 'Normal'

#     # to fix the prediction value when it is less than (Normal) ? 
#     if prediction[0][0] < 0.5:
#         prediction[0][0] = 1 - prediction[0][0]
    
#     prediction[0][0] = prediction[0][0] * 100
#     prediction = prediction[0][0].tolist()  # Convert numpy array to a Python list
#     prediction = round(prediction) 
#     # Clean up the image file if needed
#     print(prediction)
#     print("***************************************************************************")
    
#     return jsonify({'prediction': prediction, 'result': result})





















if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)



