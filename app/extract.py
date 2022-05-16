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
from db import database_connect
from reddit_connection import get_reddit_instance

# funciones que luego moveré a un archivo



if __name__ == '__main__':

    session = database_connect()

    p = Path.cwd()
    print(str(p))

    data_path = f"{str(p)}/data/reddit_account"
    log_path = f"{str(p)}/data/logs"

    parser = argparse.ArgumentParser(description='Takes submissions and redditors,'
                                                 '\n from reddit using praw and psaw,'
                                                 '\n saves them to DB')
    parser.add_argument('--option','-o', type=str, help='Modo de funcionamiento: \n[old (older posts first) | new (newer posts first)]', default='n')
    parser.add_argument('--fecha_i','-fi', type=str, help='Fecha inicio. Formato: "2012.3.23". default="2010.1.1"', default='2010.1.1')
    parser.add_argument('--fecha_f','-ff', type=str, help='Fecha final. Formato: "2013.3.23". default="2022.5.10"', default='2022.5.10')
    parser.add_argument('--subreddit','-S', type=str, help='Subreddit del que extraer publicaciones', default='cancer')

    args = parser.parse_args()
    print(args)

    credentials = f'{data_path}/client_secrets.json'

    with open(f'{log_path}/extract.txt', 'a') as f:
        f.write(f'Execution started on {dt.datetime.now()}')

    with open(credentials) as f:
        creds = json.load(f)

    reddit = get_reddit_instance(creds)

    api = PushshiftAPI(reddit)

    subreddit = args.subreddit.strip()
    gen = None

    f_inicio = args.fecha_i.strip().split(".")
    f_final = args.fecha_f.strip().split(".")

    # [date_end(2010.1.1)--------------------------date_start(datetime.now)]
    date_end = dt.datetime(int(f_inicio[0]), int(f_inicio[1]), int(f_inicio[2]))  # anho, mes, dia
    date_start = dt.datetime(int(f_final[0]), int(f_final[1]), int(f_final[2]))  # anho, mes, dia

    if args.option == 'new':
        date_begin = date_start
        date_finish = date_start - dt.timedelta(1)
    elif args.option == 'old':
        date_begin = date_end + dt.timedelta(1)
        date_finish = date_end

    subreddit = args.subreddit.strip()

    gen = None
    stop = True


    while stop:

        # use PSAW only to get id of submission in time interval
        gen = api.search_submissions(
            before=int(date_begin.timestamp()),
            after=int(date_finish.timestamp()),
            filter=['id'],
            subreddit=subreddit
        )

        # print(f'date_start: {date_start.date()} | date_begin: {date_begin.date()} \n date_end: {date_end.date()} | date_finish: {date_finish.date()}')

        for submission_id_psaw in gen:

            try:
                # use psaw here
                submission_id = submission_id_psaw

                # use praw from now on
                submission_praw = reddit.submission(id=submission_id)

                author = submission_praw.author
                if author != None and hasattr(author, 'id'):
                    with open(f'{log_path}/extract.txt', 'a') as f:
                        f.write(f'Submission created: {dt.datetime.utcfromtimestamp(submission_praw.created_utc)}\n')
                        f.write(f'Author id: {author}\n')
                        f.write(f'Submission.url: {submission_praw.url}\n')
                        f.write(40 * '*' + '\n')
                    print(f'Submission created: {dt.datetime.utcfromtimestamp(submission_praw.created_utc)}')
                    print(f'Author id: {author}')
                    print(f'Submission.url: {submission_praw.url}')
                    print(40 * '*' + '\n')

                    saveRedditor(session, reddit.redditor(author))
                    saveSubmission(session, reddit.submission(submission_praw))

                session.commit()

            except Exception as e:
                print('Error ' + str(e))
                print(traceback.format_exc())

        if args.option == 'new':
            date_begin = date_finish
            date_finish = date_finish - dt.timedelta(1)
            stop = False if date_finish == date_end else True
        elif args.option == 'old':
            date_finish = date_begin
            date_begin = date_begin + dt.timedelta(1)
            stop = False if date_begin == date_start else True
