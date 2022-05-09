from db.db_connection import database_connect
from models import Submission

# python app\example_db.py
if __name__ == "__main__":

    session = database_connect()

    submission = session.query(Submission).limit(10).all()

    print("Ten first submission....")
    for sub in submission:
        print(sub.id_submission)

    print("Search one submission....")
    search_submission = session.query(Submission).filter(Submission.id_submission == "tevgvv").first()
    print(search_submission.id_submission)