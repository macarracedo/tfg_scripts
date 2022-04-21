"""
 extract.py
 Este script hace uso de psaw para conseguir los id de las publicaciones en un intervalo de fechas indicados.
 Poseteriormente y haciendo uso de praw extrae el post completo, y guarda en la BD el submission y el redditor (autor).
"""

import datetime as dt
import traceback

import praw
import json
from psaw import PushshiftAPI
from pathlib import Path
from keeper import *
import db

p = Path.cwd()
print(str(p))

data_path = f"{str(p)}/data"

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

tf_after = int(dt.datetime(2018, 1, 1).timestamp())
tf_before = int(dt.datetime(2019, 1, 1).timestamp())
subreddit = 'cancer'

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