from builtins import Exception, eval, str
from django.views.decorators.csrf import csrf_exempt

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import MultiMatch

import json
from django.http import JsonResponse

from . import degree_plan_helper
from . import mms
from . import initialiser
from .models import Degree, PreviousStudentDegree
from .course_data_helper import track_metrics
from . import course_data_helper
from .nn import initial_network_training
from .nn import get_prediction
from .nn import train_sample
from .course_data_helper import get_all

# initialiser.initialise_database()

#initial_network_training()
print("********",len(Degree.objects.all()))
for degree in Degree.objects.all():
    degree_dict = dict()
    for course in list(map(lambda x: x['_source']['code'], get_all())):
        degree_dict[course]=0
    degree.number_of_enrolments = 1
    degree.metrics = degree_dict
    degree.save()

def all_degrees(request):
    degree_list = Degree.objects.all()
    results = []

    for degree in degree_list:
        results.append({"code": degree.code, "title": degree.name})

    return JsonResponse({"response": results})


@csrf_exempt
def degree_plan(request):
    if request.method == "GET":
        try:
            code = request.GET['query']
            starting_year = request.GET['start_year_sem']
            return degree_plan_helper.generate_degree_plan(code, starting_year)
        except(Exception):
            raise Exception("Please provide a valid degree code and starting year")
    elif request.method == "PUT":
        data = request.body.decode('utf-8')
        code = eval(data)["code"]
        courses = eval(data)["courses"]
        prev = PreviousStudentDegree(code=code, courses_taken=courses)
        prev.save()
        train_sample(Degree(code=code,title="",requirements=courses))
        track_metrics(Degree(code=code,requirements=courses))
        return JsonResponse({"response": "Success"})


def mms_request(request):
    try:
        code = request.GET['query']
        print(code)
        return mms.get_mms_data(code)
    except:
        raise Exception("Malformed JSON as input. Expects a field called query.")


def all_majors(request):
    try:
        name = request.GET['query']
        return mms.mms_by_name(name, 'majors')
    except:
        return mms.all_majors()


def all_minors(request):
    try:
        name = request.GET['query']
        return mms.mms_by_name(name, 'minors')
    except:
        return mms.all_minors()


def all_specs(request):
    try:
        name = request.GET['query']
        return mms.mms_by_name(name, 'specialisations')
    except:
        return mms.all_specs()


def course_data(request):
    try:
        query = request.GET['query']
        if query == 'titles':
            return JsonResponse({"response": course_data_helper.get_titles(request.GET.get('codes', '[]'))})
        else:
            return JsonResponse({"response": course_data_helper.get_data(query)})

    except(Exception):
        raise Exception("Please provide a valid course code")

def major_name(request):
    return

def recommend_course(request):
    plan = eval(request.GET['degree_plan'])
    code = request.GET['degree_code']
    d = Degree(code=code, requirements=plan)
    try:
        predictions = get_prediction(d)
        to_return=[]
        #get everything in elastic
        q = MultiMatch(query=code, fields=['code^4'])
        client = Elasticsearch()
        s = Search(using=client, index='courses')
        response = s.query(q).execute()
        for course in predictions:
            course_code = response[course]['hits']['hits'][0]['code']
            degree = Degree.objects.filter(code == course_code)
            proportion = int(degree.metrics[course_code])/int(degree.number_of_enrolments)
            to_return.append({"course" : course,"reason":  '%.2f \% of students in your degree took this course' % (proportion)})
        return JsonResponse({"response":to_return})
    except:
        print("network not trained, switch to search")
    return JsonResponse({"Recommendations":predictions})
