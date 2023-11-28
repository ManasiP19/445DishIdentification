from django.shortcuts import render
from django.http import HttpResponse 
from django.template import loader
from .forms import ImageForm
import requests
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess(image_path):
    img = Image.open(image_path)
    img = img.resize((299, 299))
    img_array = img_to_array(img)
    img_array /= 255.0
    if img_array.shape[-1] == 1:
        img_array = np.concatentate([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_array):
    # Load the saved model
    saved_model_path = 'C:\\Users\\manas\\445DishIdentification\\DishID\\dishIDapp\\savedModels\\googlenet.keras'
    loaded_model = load_model(saved_model_path)
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    label_encoder = LabelEncoder()
    predicted_class_label = label_encoder.classes_[predicted_class_index]
    return predicted_class_label


# Create your views here.
def mainPage(request):
    # Process images uploaded
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid(): 
            form.save()

            # Get current instance obj to display in template
            img_obj = form.instance  
            imageForModel = img_obj.image

            preprocessed_img = preprocess(imageForModel)
            prediction = predict(preprocessed_img)

            query = prediction 
            api_url = 'https://api.api-ninjas.com/v1/recipe?query={}'.format(query) 
            response = requests.get(api_url, headers={'X-Api-Key': '4AcO9BzqnfUen6lkgeBLag==pu3HsjPTjgk4VXOW'}) 
            if response.status_code == requests.codes.ok:
                apiResponse = response.json()      
            return render(request, 'mainPage.html', {'form': form, 'img_obj': img_obj, 'apiResponse': apiResponse})
    else: 
        form = ImageForm()
        return render(request, 'mainPage.html', {'form': form})
   
