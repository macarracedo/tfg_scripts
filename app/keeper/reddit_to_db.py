from sqlalchemy import select
from models import Subreddit, Redditor, Submission, Comment


def saveSubreddit(session, subreddit):
    db_subreddit = select(Subreddit.name).where(subreddit.display_name == Subreddit.name)
    resultado = session.execute(db_subreddit).fetchone()
    if resultado is not None:  # conocemos al subreddit
        print(
            "\t*** NO insertado el \"Subreddit\" con display_name: " + str(subreddit.display_name) + " *********** (Ya en la BD)")
        db_subreddit = None
    else:  # no tenemos este subreddit -> lo anhadimos
        db_subreddit = Subreddit(name=subreddit.display_name)
    return db_subreddit


def saveRedditor(session, redditor):
    result = session.query(Redditor).filter(Redditor.id_redditor == redditor.id).all()
    if result:  # conocemos al redditor
        print("\t*** NO insertado el \"Redditor\" con id:'" + str(redditor.id) + "'   \t*** (Ya en la BD)")
        db_author = None
    else:  # no tenemos este redditor -> lo anhadimos
        db_author = Redditor(id_redditor=redditor.id, name=redditor.name,
                             total_karma=redditor.total_karma, link_karma=redditor.link_karma,
                             comment_karma=redditor.comment_karma, awardee_karma=redditor.awardee_karma,
                             awarder_karma=redditor.awarder_karma, created=redditor.created,
                             created_utc=redditor.created_utc, icon_img_url=redditor.icon_img,
                             verified=redditor.verified,
                             is_blocked=redditor.is_blocked, is_employee=redditor.is_employee,
                             is_friend=redditor.is_friend,
                             is_mod=redditor.is_mod, is_gold=redditor.is_gold,
                             accept_chats=redditor.accept_chats,
                             accept_followers=redditor.accept_followers, accept_pms=redditor.accept_pms,
                             has_verified_email=redditor.has_verified_email,
                             has_subscribed=redditor.has_subscribed,
                             hide_from_robots=redditor.hide_from_robots)
        db_author = session.add(db_author)
        print("\t*** INSERTADO el \"Redditor\" con id:'" + str(redditor.id) + "'      \t***")


    return db_author


def saveSubmission(session, submission, redditor_nulo=False):
    result = session.query(Submission).filter(Submission.id_submission == submission.id).all()
    if result:  # conocemos al redditor
        print("\t*** NO insertado el \"Submission\" con id: '" + str(submission.id) + "'\t*** (Ya en la BD)")
        db_submission = None
    else:  # no tenemos este redditor -> lo anhadimos
        # NO Controlamos el autor NULO !
        query = select(Redditor.id).where(Redditor.id_redditor == submission.author.id)
        fk_id_author = session.execute(query).fetchone()[0]
        query = select(Redditor.id_redditor).where(Redditor.id_redditor == submission.author.id)
        id_author = session.execute(query).fetchone()[0]

        flair = None if submission.link_flair_text is None else submission.link_flair_text.strip()
        db_submission = Submission(id_submission=submission.id, title=submission.title,
                                   selftext=submission.selftext,
                                   fk_id_author=fk_id_author, id_author=id_author, ups=submission.ups,
                                   downs=submission.downs, upvote_ratio=submission.upvote_ratio, url=submission.url,
                                   link_flair_text=flair)  # REVISAR CORRECTO FUNCIONAMIENTO DE TRIM, PARA EVITAR DOS VALORES DE "PATIENT "
        # db_submission = db.session.add(db_submission)
        session.add(db_submission)
        print("\t*** INSERTADO el \"Submission\" con id: '" + str(submission.id) + "'   \t***")

    return db_submission



def saveComment(session, comment, submission, redditor_nulo=False):
    db_comment = select(Comment.id).where(Comment.id_comment == comment.id)
    resultado = session.execute(db_comment).fetchone()  # devuelve uno

    if resultado is not None:  # ya tenemos el comment
        # Sustituir por logger
        print("\t*** NO insertado el \"Comment\" con id: '" + str(comment.id) + "''\t*** (Ya en la BD)")
        db_comment = None
    else:  # no tenemos este comment -> lo anhadimos
        query = select(Submission.id).where(Submission.id_submission == submission.id)
        fk_id_submission = session.execute(query).fetchone()[0]

        if redditor_nulo:
            fk_id_author = 1
            id_author = 'null'
        else:
            query = select(Redditor.id).where(Redditor.id_redditor == comment.author.id)
            fk_id_author = session.execute(query).fetchone()[0]
            query = select(Redditor.id_redditor).where(Redditor.id_redditor == comment.author.id)
            id_author = session.execute(query).fetchone()[0]
        '''
            "parent_id" - The ID of the parent comment (prefixed with "t1_").
            If it is a top-level comment, this returns the submission ID instead
            (prefixed with "t3_").
        '''
        db_comment = Comment(id_comment=comment.id, fk_id_submission=fk_id_submission, id_submission=submission.id,
                             fk_id_author=fk_id_author, id_author=id_author, id_parent=comment.parent_id,
                             body=comment.body, ups=comment.ups, downs=comment.downs, depth=comment.depth)
    return db_comment

def get_submissions(session, ):
    result = session.query(Submission.link_flair_text, Submission.title, Submission.selftext).filter(
        Submission.link_flair_text != None).all()

    return result
