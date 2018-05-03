from builtins import eval

import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from .course_data_helper import get_all
from sklearn.feature_extraction.text import TfidfVectorizer
from .course_data_helper import get_data
from .models import Degree
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score


tfidf=TfidfVectorizer()
course_vectors = tfidf.fit_transform(list(map(lambda x:x['_source']['description'],get_all())))
course_ids=np.identity(len(get_all()))*100



def get_course_vector(code):
    if(get_data(code) == None):
        return None
    return tfidf.transform([get_data(code)['description']]).toarray().tolist()

def create_vector(degree_requirements):
    reqs = list(eval(degree_requirements).items())
    ave_vec = []
    for j in range(len(reqs)):
        k, v = reqs[j]
        for i in range(len(v)):
            d = v[i]
            if ("code" in d):
                if (d["code"] == "Elective Course"):
                    break
                if(get_course_vector(d["code"]) != None):
                    if ave_vec == []:
                        ave_vec = np.array(get_course_vector(d["code"])[0])
                    else:
                        ave_vec += np.array(get_course_vector(d["code"])[0])
    return np.array(ave_vec)




def create_training_arrays(degree_requirements):
    global leave_out
    leave_out = 0
    reqs = list(eval(degree_requirements).items())
    if (reqs == []):
        return (np.array([]),np.array([]))
    X = []
    Y = []
    left_out = []
    while (leave_out < 8 * len(reqs)):
        global training_vector
        training_vector = []
        ave_vec = []
        for j in range(len(reqs)):
            k, v = reqs[j]
            for i in range(len(v)):
                d = v[i]
                if ("code" in d):
                    if (d["code"] == "Elective Course"):
                        break
                    if (ave_vec == []):
                        if (leave_out == 4 * j + i + 1):
                            training_vector = d["code"]
                            left_out.append(d["code"])
                            break
                        if get_course_vector(d["code"]) != None:
                            ave_vec = np.array(get_course_vector(d["code"])[0])
                    else:
                        if (leave_out == 4 * j + i + 1):
                            training_vector = d["code"]
                            left_out.append(d["code"])
                            break
                        if get_course_vector(d["code"]) != None:
                            ave_vec += np.array(get_course_vector(d["code"])[0])
        if (not (training_vector == [])):
            if (get_data(training_vector) is not None):
                X.append(ave_vec)
                Y.append(course_ids[int(get_data(training_vector)['id'])])
        leave_out += 1
    return (np.array(X),np.array(Y))

def initial_network_training():
    network_dict = dict()
    degree_list = Degree.objects.all()
    count=0
    for degree in degree_list:

        count+=1
        if(count==10):
            break

        print(degree.requirements)
        #if degree.code is not "AACRD" or):
        #    continue

        x_array,y_array=create_training_arrays(degree.requirements)
        if(x_array.shape == (0,)) or (y_array.shape == (0,)):
            continue
        network_dict[degree.code] = MLPRegressor(hidden_layer_sizes=(1000,),activation='tanh')
        network_dict[degree.code].fit(x_array,y_array)
        test_data = list(np.argmax(network_dict[degree.code].predict(x_array), axis=1) == np.argmax(y_array, axis=1))
        print("Accuracy:", sum(test_data) / len(test_data))
    file=open("network","w")
    pickle.dump(network_dict,file)

def train_sample(degree):
    file=open("network","r")
    network_dict = pickle.load(file)
    x_array, y_array = create_training_arrays(degree.requirements)
    network_dict[degree.code].fit(x_array, y_array)
    test_data = list(np.argmax(network_dict[degree.code].predict(x_array), axis=1) == np.argmax(y_array, axis=1))
    print("Accuracy:",sum(test_data)/len(test_data))

def get_prediction(degree,number_of_recommendations):
    file = open("network", "r")
    network_dict = pickle.load(file)
    x_array = create_vector(degree.requirements)
    ids = np.argsort(network_dict[degree.code].predict(x_array))[:number_of_recommendations]
    #need to look up in elastic
    return ids