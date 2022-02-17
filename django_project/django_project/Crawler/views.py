from django.shortcuts import render
from django.http import FileResponse, HttpResponse
from django.conf import settings
import pickle
import mimetypes

# parser
import datetime
import json
import re
from bs4 import BeautifulSoup
import requests

# feat_extraction
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import tensorflow_hub as hub
# import tensorflow as tf
import nltk
from nltk.stem.snowball import SnowballStemmer
from pymorphy2 import MorphAnalyzer
from collections import Counter

# from catboost import CatBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline

import pandas as pd
from . import utils

static_path = str(settings.BASE_DIR) + '/static/'

def load_all_stuff(dir_path):
    with open(dir_path + 'catboost.pickle', 'rb') as f:
        catboost = pickle.load(f)
    with open(dir_path + 'vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(dir_path + 'classifier_model.pickle', 'rb') as f:
        clfs  = pickle.load(f)
    print(clfs)
    morph = MorphAnalyzer()
    stemmer = SnowballStemmer(language='russian')
    nlp = spacy.load('ru_core_news_sm')
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    nltk.download('stopwords')
    
    return catboost, vectorizer, nlp, model, morph, stemmer, clfs

    

def load_debug_table(static_path):
    with open(static_path + 'debug_table.pickle', 'rb') as f:
        table = pickle.load(f)
        
    return table

def save_table(static_path, table):
    with open(static_path + 'debug_table.pickle', 'wb') as f:
        pickle.dump(table, f)
        
def save_table_(static_path, table):
    with open(static_path + 'debug_table_1.pickle', 'wb') as f:
        pickle.dump(table, f)

    
    
catboost, vectorizer, nlp, model, morph, stemmer, clfs = load_all_stuff(static_path)

table = []
second_table = []
article_to_incident = {}
    
def index(request):
    global catboost, vectorizer, nlp, model, morph, stemmer, clfs, table, second_table, static_path
    context = {
        'sources': [
            {
                'name': 'znak',
                'label': 'Знак'
            },
            {
                'name': 'regnum',
                'label': 'Регнум'
            },
            {
                'name': 'interfax',
                'label': 'Интерфакс'
            },
        ]
    }
    if request.method == 'POST' and 'create_table' in request.POST:
        flag = True
        if flag:
            table = load_debug_table(static_path)
            context['table'] = [(i, len(table[i]), table[i][1:], table[i][0]) for i in range(len(table))]
            context['incident_indexes'] = [i for i in range(len(table))]
        else:
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')
            
            
            parsers = {'interfax': request.POST.get('interfax'),
                       'znak': request.POST.get('znak'),
                       'regnum' : request.POST.get('regnum')}
            
            if check_date(start_date, end_date):
                if not any(parsers.values()):
                    context['verdict'] = 'Выберите парсеры.'
                else:
                    articles = parse(parse_date(start_date), parse_date(end_date), 0.6, parsers, vectorizer, clfs, stemmer)
                    
                    if len(articles) == 0: 
                        context['verdict'] = 'За этот период статей не найдено.'
                    else:
                        table = get_clusters(articles, nlp, model, morph, stemmer, catboost)
                        save_table(static_path, table)
                            
                        context['table'] = [(i, len(table[i]), table[i][1:], table[i][0]) for i in range(len(table))]
                        
                        context['incident_indexes'] = [i for i in range(len(table))]
            else:
                context['verdict'] = 'Одна или обе даты заданы некорректно, проверьте.'
                
    if request.method == 'POST' and 'accept_result' in request.POST:
        changes = {}
        
        for i in range(len(table)):
            if request.POST.get('delete ' + str(i)):
                table[i] = []
                
        all_articles = []
        for elem in table:
            all_articles += elem
            
        for elem in all_articles:
            option = request.POST.get(str(elem[0]))
            if option:
                if len(option.split()[-1]) == 1:
                    changes[elem[0]] = int(option.split()[-1])
                else:
                    changes[elem[0]] = -1
            
        table = redraw_table(changes, all_articles)
        second_table = create_second_table(table)
        context['table'] = [(i, len(table[i]), table[i][1:], table[i][0]) for i in range(len(table))]

        context['second_table'] = [(i, len(second_table[i][3]), second_table[i][0], second_table[i][1], second_table[i][2],
                                    second_table[i][3][0], second_table[i][3][1:]) for i in range(len(second_table))]
        
        context['incident_indexes'] = [i for i in range(len(table))]
        
    if request.method == 'POST' and 'create_incident' in request.POST:
        empty_article = (-1, '', '', '')
        if table[-1][0] == empty_article:
            context['verdict'] = 'Уже создан пустой инцидент.'
            context['table'] = [(i, len(table[i]), table[i][1:], table[i][0]) for i in range(len(table))]
            context['second_table'] = [(i, len(second_table[i][3]), second_table[i][0], second_table[i][1], second_table[i][2],
                                    second_table[i][3][0], second_table[i][3][1:]) for i in range(len(second_table))]
            context['incident_indexes'] = [i for i in range(len(table))]
            
        else:
            table.append([empty_article])
            context['table'] = [(i, len(table[i]), table[i][1:], table[i][0]) for i in range(len(table))]
            context['second_table'] = [(i, len(second_table[i][3]), second_table[i][0], second_table[i][1], second_table[i][2],
                                    second_table[i][3][0], second_table[i][3][1:]) for i in range(len(second_table))]
            context['incident_indexes'] = [i for i in range(len(table))]
            
    if request.method == 'POST' and 'accept_second_result' in request.POST:
        for i in range(len(second_table)):
            loc = request.POST.get('loc ' + str(i))
            org = request.POST.get('org ' + str(i))
            type_ = request.POST.get('type ' + str(i))
            print(loc, org, type_) 
            incident = (second_table[i][0], second_table[i][1], [loc, org, type_], second_table[i][3])
            second_table[i] = incident
            
        context['table'] = [(i, len(table[i]), table[i][1:], table[i][0]) for i in range(len(table))]

        context['second_table'] = [(i, len(second_table[i][3]), second_table[i][0], second_table[i][1], second_table[i][2],
                                    second_table[i][3][0], second_table[i][3][1:]) for i in range(len(second_table))]
        
        context['incident_indexes'] = [i for i in range(len(table))]    
        save_table_(static_path, second_table)    
            
    return render(request, 'index.html', context)

def generate_report(static_path, table):
    parsers = {'www.interfax.ru': get_parsed_interfax_news_report,
           'www.znak.com': get_parsed_znak_news_report,
           'regnum.ru': get_parsed_regnum_news_report
         }
    df = make_dataframe(table, parsers)
    print(df)
    df.to_excel(static_path + 'report.xlsx')
    
    
def download_file(request):
    global static_path, table
    generate_report(static_path, table)
    filename = 'report.xlsx'
    filepath = static_path + filename
    path = open(filepath, 'rb')
    mime_type, _ = mimetypes.guess_type(filepath)
    response = HttpResponse(path, content_type=mime_type)
    response['Content-Disposition'] = "attachment; filename=%s" % filename
    return response

def get_parsed_interfax_news_report(url):
    doc = ''
    soup = ''
    try:
        doc = requests.get(url)
        soup = BeautifulSoup(doc.content, 'html.parser')
    except:
        return None, None
    header = soup.find('h1').text
    text = []
    for p in soup.find_all('p'):
        text.append(p.text)
    text = ''.join(text)
    return header, text 

def get_parsed_regnum_news_report(url):
    header = ''
    news_link = url
    soup = ''
    r = ''
    try:
        r = requests.get(news_link)
        soup = BeautifulSoup(r.text, 'html.parser')
    except:
        return None, None
    doc = soup.findAll('p')
    res = []
    for d in doc:
        text = re.findall(r'[А-ЯЁ]+.*?\.', str(d))
        res.append(text)
    txt = ''
    for r in res:
        if len(r) > 0:
            if 'гиперссылка на ИА REGNUM' in r[0]:
                break
            txt += r[0]
    return header, txt

def get_parsed_znak_news_report(url):
    try:
        doc = requests.get(url)
    except:
        return None, None
    soup = BeautifulSoup(doc.text, 'html.parser')
    header = soup.find('h1').text
    text = []
    for p in soup.find_all('p'):
        text.append(p.text)
    text = ''.join(text)
    return header, text



def parse(start_date, end_date, threshold, parsers, vectorizer, clfs, stemmer):
    results = []
    if parsers['regnum']:
        results += utils.parse_regnum(start_date, end_date, threshold, vectorizer, clfs, stemmer)
    print(len(results))
    
    if parsers['interfax']:
        results += utils.parse_interfax(start_date, end_date, threshold, vectorizer, clfs, stemmer)
    print(len(results))
    
    if parsers['znak']:
        results += utils.parse_znak(start_date, end_date, threshold, vectorizer, clfs, stemmer)
    print(len(results))
    return results

def get_clusters(articles, nlp, model, morph, stemmer, catboost):
    texts = []
    headers = []
    dates = []
    urls = []
    
    for article in articles:
        headers.append(article[0])
        texts.append(article[1])
        dates.append(article[2])
        urls.append(article[3])
    
    prepared_pairs = utils.prepare_data(headers, texts, model, nlp, morph, stemmer)
    predicted_edges = {}
    for key in prepared_pairs.keys():
        predicted_edges[key] = catboost.predict_proba([prepared_pairs[key]])[0][1]
    
    min_prob = 0.99

    graph = {}
    for pair in predicted_edges:
        if predicted_edges[pair] > min_prob:
            u = pair[0]
            v = pair[1]
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
                
            graph[u].append(v)
            graph[v].append(u)
    
    clusters = []
    components = utils.make_components(graph)
    for component in components:
        cluster = []
        for index in list(component.keys()):
            if 'interfax' in urls[index]:
                source = 'Интерфакс'
            elif 'regnum' in urls[index]:
                source = 'Регнум'
            else:
                source = 'Знак'
            cluster.append((index, urls[index], source, headers[index], dates[index]))
        clusters.append(cluster)

    return clusters
    
def redraw_table(changes, all_articles):
    new_table = []
    incidents_count = max(changes.values()) + 1
    for i in range(incidents_count):
        new_table.append([])
    for key in changes.keys():
        for article in all_articles:
            if key == article[0] and changes[key] >=0:
                new_table[changes[key]].append(article)
    new_table = delete_empty_incidents(new_table)
    
    return new_table

def delete_empty_incidents(table):
    new_table = []
    for incident in table:
        if (-1, '', '', '') in incident:
            if len(incident) > 1:
                new_incident = []
                for article in incident:
                    if article != (-1, '', '', ''):                        
                        new_incident.append(article)                
                new_table.append(new_incident) 
        elif incident != []:
            new_table.append(incident)
    return new_table
            
def parse_date(date):
    return datetime.date.fromisoformat(date)


def check_date(start, end):
    if start == None or end == None:
        return False
    too_past = datetime.datetime.strptime('01-01-2000', '%d-%m-%Y')
    too_future = datetime.datetime.today()
    try:
        start_date = parse_date(start)
        end_date = parse_date(end)
    except:
        return False

    if start_date > end_date or start_date < too_past or start_date > too_future or end_date < too_past or end_date > too_future:
        return False
    return True

def create_second_table(first_table):
    parsers = {'www.interfax.ru': get_parsed_interfax_news_report,
           'www.znak.com': get_parsed_znak_news_report,
           'regnum.ru': get_parsed_regnum_news_report
    }
    second_table = []
    for incident in first_table:
        new_incident = create_incident(incident, parsers)
        second_table.append(new_incident)
    return second_table    
             
def create_incident(incident, parsers):
    global nlp, morph
    new_articles = []
    for article in incident:
        num, url, source, header, date = article
        new_articles.append((header, url))
        current_orgs = Counter()
        current_locs = Counter()
        name = url.split('/')[2]
        if name in parsers:
            parse_func = parsers[name]
            _, text = parse_func(url)
            ents = utils.get_ners_with_types(text, nlp, morph)
            for ent in ents:
                if ent[1] == 'ORG':
                    if ent[0] not in current_orgs:
                        current_orgs[ent[0]] = 0
                    current_orgs[ent[0]] += 1
                if ent[1] == 'LOC':
                    if ent[0] not in current_locs:
                        current_locs[ent[0]] = 0
                    current_locs[ent[0]] += 1
    predicted_locs = ', '.join([ent[0] for ent in current_locs.most_common(3)])
    predicted_orgs = ', '.join([ent[0] for ent in current_orgs.most_common(3)])
    first_previous = ['', '', '']
    new_incident = (predicted_locs, predicted_orgs, first_previous, new_articles)
    return new_incident
           
        

def make_dataframe(incidents, parsers):
    global nlp, morph
    k = 0
    names = []
    headers = []
    links = []
    dates = []
    orgs = []
    locations = []
    for incident in incidents:
        names += [f'Инцидент {k}']
        k += 1
        current_orgs = set()
        current_locations = set()
        for article in incident:
            num, url, header, date = article
            headers.append(header)
            links.append(url)
            dates.append(date)
            name = url.split('/')[2]
            if name in parsers:
                parse_func = parsers[name]
                header, text = parse_func(url)
                #print(url, header, text, name)
                ents = utils.get_ners_with_types(text, nlp, morph)
                for ent in ents:
                    if ent[1] == 'ORG':
                        current_orgs.add(ent[0])
                    if ent[1] == 'LOC':
                        current_locations.add(ent[0])
        orgs += list(current_orgs)
        locations += list(current_locations)
        max_len = max(len(names), len(headers), len(orgs), len(locations))
        if len(names) < max_len:
            for _ in range(max_len - len(names)):
                names.append('')
        if len(headers) < max_len:
            for _ in range(max_len - len(headers)):
                headers.append('')
        if len(links) < max_len:
            for _ in range(max_len - len(links)):
                links.append('')
        if len(dates) < max_len:
            for _ in range(max_len - len(dates)):
                dates.append('')
        if len(orgs) < max_len:
            for _ in range(max_len - len(orgs)):
                orgs.append('')
        if len(locations) < max_len:
            for _ in range(max_len - len(locations)):
                locations.append('')
                
    final_data = {'Название': names,
                  'Заголовок': headers,
                  'Ссылка': links,
                  'Дата': dates,
                  'Организации': orgs,
                  'Локации': locations}
    
    df = pd.DataFrame(data=final_data)
    
    return df
