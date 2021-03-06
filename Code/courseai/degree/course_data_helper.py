import json
from builtins import str, eval

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import MultiMatch


def get_data(code):
    q = MultiMatch(query=code, fields=['code^4'])
    client = Elasticsearch()
    s = Search(using=client, index='courses')
    response = s.query(q).execute()

    try:
        hit = response['hits']['hits'][0]
    except IndexError:
        return None
    course_data = {"course_code": hit['_source']['code'],
                   "id": hit["_id"],
                   "title": hit['_source']['title'],
                   "description": hit['_source']['description'],
                   "learning_outcomes": hit['_source']['outcome'],
                   "prerequisite_text": hit['_source']['prereq_text'],
                   "prerequisites": eval(str(hit['_source']['pre_req_cnf'])),
                   "semester": eval(str(hit['_source']['semester']))
                   }
    return course_data


def track_metrics(degree_plan):
    return


def get_titles(codes):
    course_data = []
    codes = json.loads(codes)
    for code in codes:

        q = MultiMatch(query=code, fields=['code^4'])
        client = Elasticsearch()
        s = Search(using=client, index='courses')
        response = s.query(q).execute()

        try:
            hit = response['hits']['hits'][0]
        except IndexError:
            continue

        course_data.append({"course_code": hit['_source']['code'],
                            "title": hit['_source']['title']})
    return course_data


def get_prereqs(codes):
    course_data = {}
    codes = json.loads(codes)
    for code in codes:

        q = MultiMatch(query=code, fields=['code^4'])
        client = Elasticsearch()
        s = Search(using=client, index='courses')
        response = s.query(q).execute()

        try:
            hit = response['hits']['hits'][0]
        except IndexError:
            continue

        course_data[hit['_source']['code']] = {"course_code": hit['_source']['code'],
                                               "prerequisite_text": hit['_source']['prereq_text'],
                                               "prerequisites": eval(str(hit['_source']['pre_req_cnf'])),
                                               "semester": eval(str(hit['_source']['semester']))}
        pass
    return course_data


def get_all():
    client = Elasticsearch()
    s = Search(using=client, index='courses')
    count = s.count()
    result = s[0:count].execute()['hits']['hits']
    result = sorted(result, key=lambda x: int(x["_id"]))
    return result
