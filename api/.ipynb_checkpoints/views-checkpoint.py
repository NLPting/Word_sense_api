from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
import json
from .wsd_bert import *



# Create your views here.
    
@csrf_exempt
def word_sense(request):
    if request.method == "POST":
        if request.body:
            json_data = json.loads(request.body)
            print(json_data)
            sentence = json_data["Sentence"]
            main_word = json_data["MainWord"]
            word_level = wsd_level(sentence , main_word)
            print(word_level)
            
            return JsonResponse({"Level":word_level , "Word" : main_word})
        else:
            return HttpResponse("Your request body is empty.")
    else:
        return HttpResponse("Please use POST.")
    
    
