from sklearn.feature_extraction.text import TfidfVectorizer
# from degree import course_data_helper

import operator

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import MultiMatch
from degree.course_data_helper import get_all


tfidf=TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english', max_df=0.7)
tfidf.fit(list(map(lambda x:x['_source']['description'],get_all())))

def raw_search(search_object, phrase, codes, levels):
    q = MultiMatch(query=phrase, fields=['title^2', 'description', 'outcome^1.5'])

    response = {}

    if codes is None and levels is None:
        response = search_object.query(q).execute()

    else:
        print("You're calling the wrong function")

    course_list = []
    for hit in response['hits']['hits']:
        course = {'code': hit['_source']['code'], 'title': hit['_source']['title']}
        course_list.append(course)
    return course_list

def get_data(code):
    q = MultiMatch(query=code, fields=['code'])
    client = Elasticsearch()
    s = Search(using=client, index='courses')
    response = s.query(q).execute()

    try:
        hit = response['hits']['hits'][0]
    except IndexError:
        return None

    course_data = {
                   "description": hit['_source']['description'],
                   }
    return course_data


def get_descriptions(course_list):

    descriptions = []

    for course in course_list:
        desc = get_data(course)["description"]
        descriptions.append(desc)

    return descriptions


def get_recommendations(course_list):
    course_descriptions = get_descriptions(course_list)

    tfidf_matrix = tfidf.transform(course_descriptions)
    feature_names = tfidf.get_feature_names()
    dense = tfidf_matrix.todense()

    tags = {}

    for idx in range(len(course_list)):
        episode = dense[idx].tolist()[0]
        phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]

        z = sorted(phrase_scores, key=lambda t: t[1] * -1)[:10]
        features = []
        for i, s in z:
            features.append((feature_names[i], s))
            if feature_names[i] not in tags:
                tags[feature_names[i]] = s
            else:
                tags[feature_names[i]] += s

    sorted_tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)[:10]

    client = Elasticsearch()
    s = Search(using=client, index='courses')
    recommendations = []

    for tag in sorted_tags:
        # search on the search engine
        courses = raw_search(s, '"' + tag[0] + '"', codes=None, levels=None)
        course_codes = [course['code'] for course in courses]

        # print(course_codes)

        # get the top result that has not been done
        for code in course_codes:
            if code in course_list:
                continue
            else:
                recommendations.append(code)
                break

    return recommendations

