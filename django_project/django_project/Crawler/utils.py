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
import tensorflow as tf
import nltk
from nltk.stem.snowball import SnowballStemmer
from pymorphy2 import MorphAnalyzer

from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd


def get_dates_between(start_date, end_date):
    if start_date > end_date:
        return []
    dates_between = []
    dates = pd.date_range(start_date, end_date).tolist()
    for date in dates:
        dates_between.append(date.date())
    return dates_between

def parse_interfax(start_date, end_date, threshold, vectorizer, clfs, stemmer):
    parse_results = []
    base_url = 'https://www.interfax.ru/news/'
    dates = get_dates_between(start_date, end_date) # Example https://www.interfax.ru/news/2022/01/24
    for date in dates:
        date_str = str(date).split('-')
        link = base_url + date_str[0] + '/' + date_str[1] + '/' + date_str[2]
        doc = requests.get(link)
        soup = BeautifulSoup(doc.text, 'html.parser')
        for a in soup.find_all('a'):
            try:
                ref = a.attrs['href']
                if len(ref.split('/')[2]) == 6:
                    header, text = get_parsed_interfax_news(base_url[:-6] + ref, threshold, vectorizer, clfs, stemmer)
                    if header:
                        parse_results.append((header, text, str(date), base_url[:-6] + ref))
            except: pass
    return parse_results

def get_parsed_interfax_news(url, threshold, vectorizer, clfs, stemmer):
    doc = requests.get(url)
    soup = BeautifulSoup(doc.content, 'html.parser')
    header = soup.find('h1').text
    text = []
    for p in soup.find_all('p'):
        text.append(p.text)
    text = ''.join(text)
    if predict_proba(text, vectorizer, clfs, stemmer) > threshold:
        return header, text
    return None, None

def parse_znak(start_date, end_date, threshold, vectorizer, clfs, stemmer):
    parse_results = []
    base_url = 'https://www.znak.com/'
    dates = get_dates_between(start_date, end_date) # Example https://www.znak.com/2022-01-12/
    for date in dates:
        
        link = base_url + str(date) + '/'
        try:
            doc = requests.get(link)
        except:
            continue
        soup = BeautifulSoup(doc.text, 'html.parser')
        last_data_ts = ''
        for a in soup.find_all('a', {'class': 'pub'}):
            time = a.find('time').attrs['datetime'][:10]
            last_data_ts = a.attrs['data-ts']
            if time == str(date):
                header, text = get_parsed_znak_news(base_url + a.attrs['href'], threshold, vectorizer, clfs, stemmer)
                if header:
                    parse_results.append((header, text, str(date), base_url[:-1] + a.attrs['href']))
                    
        
        if last_data_ts:
            link = f'https://www.znak.com/ajax/older/{last_data_ts}/500'
            try:
                doc = requests.get(link)
                soup = BeautifulSoup(doc.text, 'html.parser')
            except:
                pass
            for a in soup.find_all('a'):
                time = a.find('time').attrs['datetime'][:10]
                if time == str(date):
                    header, text = get_parsed_znak_news(base_url + a.attrs['href'], threshold, vectorizer, clfs, stemmer)
                    if header:
                        #print(base_url[:-1] + a.attrs['href'], type(base_url[:-1] + a.attrs['href']) , type(base_url + a.attrs['href']))
                        parse_results.append((header, text, str(date), base_url[:-1] + a.attrs['href']))
                    
                else:
                    break
    
    return parse_results
        
def get_parsed_znak_news(url, threshold, vectorizer, clfs, stemmer):
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
    if predict_proba(text, vectorizer, clfs, stemmer) > threshold:
        return header, text
    return None, None


def parse_regnum(start_date, end_date, threshold, vectorizer, clfs, stemmer):
    link = 'https://regnum.ru/api/get/search/all?q='
    dates = get_dates_between(start_date, end_date)
    res_texts = []
    for date in dates:
        day = date.day
        month = date.month
        if date.day < 10:
            day = '0' + str(day)
        if month < 10:
            month = '0' + str(month)
        
        for i in range(1, 10):
            cur_link = f'{link}{date.year}-{month}-{day}&page={i}&filter={{"authorId":"","regionsId":"","theme":""}}'
            r = ''
            try:
                r = requests.get(cur_link)
            except: 
                print('1')
            if not r:
                continue
            data = json.loads(r.text)
            if not data['articles']:
                break
            for article in data['articles']:
                header = article['news_header']
                news_link = article['news_link']
                soup = ''
                try:
                    r = requests.get(news_link)
                    soup = BeautifulSoup(r.text, 'html.parser')
                except Exception as e: 
                    continue
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
                if predict_proba(txt, vectorizer, clfs, stemmer) > threshold:   
                    res_texts.append((header, txt, str(date), news_link))
                
    return res_texts

def clean_text(text, stemmer):
    t = text.lower()
    t = re.findall(r"[а-я]+|[.,!?;]", t)
    stopwords = nltk.corpus.stopwords.words('russian')
    stopwords += ['http', 'com', 'www', 'org']
    filter_words = []
    for w in t:
        if w not in stopwords:
            filter_words.append(w)
    res = []
    stemm_words = []
    for w in filter_words:
        stemm_words.append(stemmer.stem(w))
    for w in stemm_words:
        if 1 < len(w) <= 20:
            res.append(w)
    return ' '.join(res)


def predict_proba(text, vectorizer, clfs, stemmer):
    text = clean_text(text, stemmer)
    
    vec_text = vectorizer.transform([text])[0]
    
    predictions = []
    
    for clf in clfs:
        try:
            y_pred = clf.predict_proba(vec_text)
        except:
            X_t = vec_text.todense()
            y_pred = clf.predict_proba(X_t)
        predictions.append(y_pred)

    res_proba = []

    for p in predictions:
        if len(res_proba) == 0:
            res_proba = p
        else:
            res_proba += p
    res_proba /= 5
    
    return res_proba[0][1]


def get_clean_texts(texts):
    clean_data = []
    for text in texts:
        clean_text = re.findall(r'[А-ЯЁ]+.*?\.', text)
        clean_text = ' '.join(clean_text)
        if len(clean_text) > 0:
            clean_data.append(clean_text)
    return clean_data

def get_embeddings(texts, model):
    all_embeddings = []
    for text in texts:
        embedding = model([text])
        all_embeddings.append(embedding.numpy())
    return all_embeddings

def get_dict(text, stemmer):
    cur_dict = set()
    text = text.lower()
    text = re.findall(r"[а-я]+|[.,!?;]", text)
    stopwords = nltk.corpus.stopwords.words('russian')
    stopwords += ['http', 'com', 'www', 'org']
    filter_words = []
    for word in text:
        if word not in stopwords:
            filter_words.append(word)
    result = []
    stemm_words = []
    for word in filter_words:
        stemm_words.append(stemmer.stem(word))
    for word in stemm_words:
        if 1 < len(word) <= 20:
            result.append(word)
    return set(result)

def get_ners_with_types(text, nlp, morph):
        ners = set()
        doc = nlp(text)
        for ent in doc.ents:
            ner = (lemmitize_ent(ent.text, morph), ent.label_)
            ners.add(ner)
        return list(set(ners))

def lemmitize_ent(ent, morph):
    lemmitize_ent = []
    for word in ent.split():
        lemmitize_ent.append(morph.normal_forms(word)[0])
        
    return ' '.join(lemmitize_ent)
    
def tf_idf_vectorize(clean_texts):
    vectorizer_1 = TfidfVectorizer(max_features=3000, ngram_range=(1, 3))
    vectorizer_2 = TfidfVectorizer(max_features=3000)
    vectorizer_3 = TfidfVectorizer(max_features=3000, ngram_range=(2, 2))
    vectorizer_4 = TfidfVectorizer(max_features=3000, ngram_range=(3, 3))
    X_1 = vectorizer_1.fit_transform(clean_texts)
    X_2 = vectorizer_2.fit_transform(clean_texts)
    X_3 = vectorizer_3.fit_transform(clean_texts)
    X_4 = vectorizer_4.fit_transform(clean_texts)

    all_tfidf = []

    for text in clean_texts:
        emb_X_1 = vectorizer_1.transform([text])
        emb_X_2 = vectorizer_2.transform([text])
        emb_X_3 = vectorizer_3.transform([text])
        emb_X_4 = vectorizer_4.transform([text])
        all_tfidf.append([emb_X_1, emb_X_2, emb_X_3, emb_X_4])
    return all_tfidf


def get_proba_by_dict(dict_, other_dict):
    dict_intersection = set()
    dict_intersection = dict_.intersection(other_dict)
    p = (len(dict_intersection)) / (min(len(dict_), len(other_dict)) + 1e-7)
    return p

def get_proba_by_ners_org(ners, other_ners):
    ner_intersection = set()
    ners_org = set([elem[0] for elem in ners if elem[1] == 'ORG'])
    other_ners_org = set([elem[0] for elem in other_ners if elem[1] == 'ORG'])
    ner_intersection = ners_org.intersection(other_ners_org)
    p = (len(ner_intersection)) / (min(len(ners_org), len(other_ners_org)) + 1e-7)
    return p

def get_proba_by_ners_loc(ners, other_ners):
    ner_intersection = set()
    ners_loc = set([elem[0] for elem in ners if elem[1] == 'LOC'])
    other_ners_loc = set([elem[0] for elem in other_ners if elem[1] == 'LOC'])
    ner_intersection = ners_loc.intersection(other_ners_loc)
    p = (len(ner_intersection)) / (min(len(ners_loc), len(other_ners_loc)) + 1e-7)
    return p
    
def get_proba_by_embeddings(emb, other_emb):
    try:
        p = cosine_similarity(emb, other_emb)[0][0]
        return p
    except:
        p = cosine_similarity(emb[0], other_emb[0])[0][0]
        return p

def get_both_probas_label(transformed_data, other_transformed_data):
    proba_ners = get_proba_by_ners(transformed_data[0], other_transformed_data[0])
    proba_embs = get_proba_by_embeddings(transformed_data[1], other_transformed_data[1])
    if transformed_data[2] == other_transformed_data[2]:
        label = 1
    else: 
        label = 0
    return (proba_ners, proba_embs, label)

def get_pair_features(text_features, other_text_features):
    p_emb_head = get_proba_by_embeddings(text_features[0], other_text_features[0])
    p_org = get_proba_by_ners_org(text_features[1], other_text_features[1])
    p_loc = get_proba_by_ners_loc(text_features[1], other_text_features[1])
    p_dict = get_proba_by_dict(text_features[2], other_text_features[2])
    p_emb_tf_idf_1 = get_proba_by_embeddings(text_features[3], other_text_features[3])
    p_emb_tf_idf_2 = get_proba_by_embeddings(text_features[4], other_text_features[4])
    p_emb_tf_idf_3 = get_proba_by_embeddings(text_features[5], other_text_features[5])
    p_emb_tf_idf_4 = get_proba_by_embeddings(text_features[6], other_text_features[6])
    return (p_emb_head, p_org, p_loc, p_dict, p_emb_tf_idf_1, p_emb_tf_idf_2, p_emb_tf_idf_3, p_emb_tf_idf_4)

def prepare_data(headers, texts, model, nlp, morph, stemmer):
    clean_texts = get_clean_texts(texts)
    tf_idf_vectors = tf_idf_vectorize(clean_texts)
    prepared_data = []
    for header, text, tf_idf_vectors in zip(headers, clean_texts, tf_idf_vectors):
        prepared_data.append((get_embeddings(header, model), get_ners_with_types(text, nlp, morph), get_dict(text, stemmer), tf_idf_vectors[0],                              tf_idf_vectors[1], tf_idf_vectors[2], tf_idf_vectors[3]))
    prepared_pairwise_data = {}
    for i in range(len(prepared_data)):
        for j in range(len(prepared_data)):
            if (j, i) not in prepared_pairwise_data.keys() and j != i:
                prepared_pairwise_data[(i, j)] = get_pair_features(prepared_data[i], prepared_data[j])
    return prepared_pairwise_data

def get_component(graph, v):
    visited = {}
    stack = []
    stack.append(v)
    while stack:
        cur = stack.pop()
        visited[cur] = graph[cur]
        for u in graph[cur]:
            if u not in visited:
                stack.append(u)
                visited[u] = graph[u]
    return visited

def make_components(graph):
    visited = []
    components = []
    for i in graph:
        cur = []
        if i not in visited:
            cur = get_component(graph, i)
            visited += cur.keys()
            components.append(cur)
    return components

