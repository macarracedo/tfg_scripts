"""
 extract.py
 Este script hace uso de psaw para conseguir los id de las publicaciones en un intervalo de fechas indicados.
 Poseteriormente y haciendo uso de praw extrae el post completo, y guarda en la BD el submission y el redditor (autor).
"""

import datetime as dt
import traceback
import argparse
import praw
import json
from psaw import PushshiftAPI
from pathlib import Path
from keeper import *
import db

p = Path.cwd()
print(str(p))

filename = f"{str(p)}/data/prep_tf-idf.p"
data_path = f"{str(p)}/data"

parser = argparse.ArgumentParser(description='Takes submissions and redditors,'
                                             '\n from reddit using praw and psaw,'
                                             '\n saves them to DB')
parser.add_argument('--fecha_i','-fecha_i', type=str, help='Fecha inicio. Formato: "2012.3.23"', default='2020.1.1')
parser.add_argument('--fecha_f','-fecha_f', type=str, help='Fecha final. Formato: "2013.3.23"', default='2021.1.1')
parser.add_argument('--subreddit','-S', type=str, help='Subreddit del que extraer publicaciones', default='2021.1.1')


parser.add_argument('--clean_text', nargs='?', const=False, type=bool, default=True, help="Don't use clean_text library")
parser.add_argument('--tokenize', nargs='?', const=True, type=bool, default=False, help='Tokenize result column')
parser.add_argument('--stopwords', nargs='?', const=False, type=bool, default=True, help="Don't remove stopwords")


group_args = parser.add_mutually_exclusive_group()
group_args.add_argument('-l', '--lemmatization', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palablas a su lema')
group_args.add_argument('-s', '--stemming', nargs='?', const=True, type=bool, default=False,
                        help='Reduce las palabras a su raiz')

args = parser.parse_args()
print(args)


"""
    reddit.validate_on_submit = True
    enable_commit = True
    enable_submission_insert = True
    enable_comment_insert = True
    enable_redditor_insert = True
    post_limit = 50000
    comment_depth_limit = 4
"""

credentials = f'{data_path}/client_secrets.json'

with open(credentials) as f:
    creds = json.load(f)

# Config Reddit connection
reddit = praw.Reddit(client_id=creds['client_id'],
                     client_secret=creds['client_secret'],
                     user_agent=creds['user_agent'],
                     redirect_uri=creds['redirect_uri'],
                     refresh_token=creds['refresh_token'])
api = PushshiftAPI(reddit)


f_inicio = args.fecha_i.strip().split(".")
f_final = args.fecha_f.strip().split(".")

# anho, mes, dia
tf_after = int(dt.datetime(int(f_inicio[0]), int(f_inicio[1]), int(f_inicio[2])).timestamp())
tf_before = int(dt.datetime(int(f_final[0]), int(f_final[1]), int(f_final[2])).timestamp())
subreddit = args.subreddit.strip()

# use PSAW only to get id of submission in time interval
gen = api.search_submissions(
    after=tf_after,
    before=tf_before,
    filter=['id'],
    subreddit=subreddit
)

for submission_id_psaw in gen:
    try:
        # use psaw here
        submission_id = submission_id_psaw

        # use praw from now on
        submission_praw = reddit.submission(id=submission_id)

        author = submission_praw.author
        if author != None and hasattr(author, 'id'):
            print(f'author id: {author}')
            print(f'submission id: {submission_praw.id}\n \tselftext: {submission_praw.selftext}')
            #print(str(saveRedditor(reddit.redditor(author))))
            saveRedditor(reddit.redditor(author))
            #print('type(submission_praw): ' + str(type(submission_praw)))
            #print('type(reddit.submission(submission_praw)): ' + str(type(reddit.submission(submission_praw))))
            saveSubmission(reddit.submission(submission_praw))

        db.session.commit()

    except Exception as e:
        print('Error ' + str(e))
        print(traceback.format_exc())