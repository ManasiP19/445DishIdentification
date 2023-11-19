from django.db import models
from django.form import fields
from .models import Image
from django import forms


class userImage(forms.ModelForm):
    class meta: 
        models = Image
        fields = '__all__'

        