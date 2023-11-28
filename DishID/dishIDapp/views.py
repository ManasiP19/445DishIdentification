from django.shortcuts import render
from django.http import HttpResponse 
from django.template import loader
from .forms import ImageForm
import requests
from tensorflow.keras.models import load_model

# Load the saved model
saved_model_path = 'C:\\Users\\mrswi\\OneDrive\\Documents\\Penn State\\Machine Learning\\Final.FinalProject\\445DishIdentification\\DishID\\dishIDapp\\savedModels\\googlenet.keras'
loaded_model = load_model(saved_model_path)

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

            prediction = load_model.predict(imageForModel)

            query = prediction 
            api_url = 'https://api.api-ninjas.com/v1/recipe?query={}'.format(query) 
            response = requests.get(api_url, headers={'X-Api-Key': '4AcO9BzqnfUen6lkgeBLag==pu3HsjPTjgk4VXOW'}) 
            if response.status_code == requests.codes.ok:
                apiResponse = response.json()      
            return render(request, 'mainPage.html', {'form': form, 'img_obj': img_obj, 'apiResponse': apiResponse})
    else: 
        form = ImageForm()
        return render(request, 'mainPage.html', {'form': form})
   
