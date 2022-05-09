from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def database_connect():

    engine = create_engine(
        "mysql+pymysql://redditUser:redditPass@localhost:3306/tfg_test?charset=utf8mb4"
    )

    Session = sessionmaker(bind=engine)

    return Session()