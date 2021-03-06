from django.http import HttpResponse
from django.template import loader

from degree_builder import recommender
from .models import Student


# Create your views here.

def index(request):
    return HttpResponse(loader.get_template('dynamic_pages/index.html').render())


def planner(request):
    template = loader.get_template('dynamic_pages/planner.html')
    context = {
        'degree_name': request.GET['degreeName'],
        'degree_code': request.GET['degreeCode'],
        'start_year': request.GET['startyear'],
        'start_sem': request.GET['semester']
    }
    return HttpResponse(template.render(context))


def user_login(request):
    # if reached here from home page
    if 'uname' in request.POST and 'year' not in request.POST:
        username = request.POST['uname']
        template = loader.get_template('dynamic_pages/degree_builder.html')

        # if there is no existing entry in the database, add the person to the database and redirect them to the degree page
        try:
            student = Student.objects.filter(name=username).get()

            context = {
                'username': username,
                'start_year': student.start_year,
                'start_semester': student.start_semester,
                'interests': student.interests,
                'degree': student.degree,
                'recommended_courses': recommender.get_recommendations(student.interests, student.degree)
            }

        except Student.DoesNotExist:

            Student.objects.create(name=username)

            context = {
                'username': username,
                'start_year': "",
                'start_semester': "",
                'interests': "",
                'degree': "",
                'recommended_courses': ""
            }

        return HttpResponse(template.render(context, request))

    if 'uname' in request.POST and 'year' in request.POST:
        # save data in database
        username = request.POST['uname']
        template = loader.get_template('dynamic_pages/degree_builder.html')

        start_year = request.POST['year']
        start_semester = request.POST['semester']
        interests = request.POST['interests']
        degree = request.POST['degree']

        Student.objects.filter(name=username).update(start_year=start_year, start_semester=start_semester,
                                                     interests=interests, degree=degree)

        context = {
            'username': username,
            'start_year': start_year,
            'start_semester': start_semester,
            'interests': interests,
            'degree': degree,
            'recommended_courses': recommender.get_recommendations(interests, degree)
        }

        return HttpResponse(template.render(context, request))

    raise NotImplementedError("Page seems to have been refreshed or something")


def get_degree(request):
    raise NotImplementedError("Get degree not implemented")


def update_degree(request):
    raise NotImplementedError("Update degree not implemented")
