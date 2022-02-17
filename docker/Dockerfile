FROM python

WORKDIR /apps/django_project

COPY ./django_project /apps/django_project

COPY ./nltk_data /root/nltk_data

EXPOSE 55051

RUN pip install -r requirements.txt

CMD ["python", "/apps/django_project/manage.py", "runserver", "0.0.0.0:55051"]

