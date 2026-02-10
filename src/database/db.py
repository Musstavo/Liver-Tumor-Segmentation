import os
from sqlmodel import SQLModel, create_engine, Session

database_url = os.getenv(
    "DATABASE_URL", "postgresql://lmao_admin:lmao_pass@localhost:5432/liver_db"
)

if "db:5432" in database_url and os.getenv("DOCKER_ENV") is None:
    database_url = database_url.replace("@db:5432", "@localhost:5432")

engine = create_engine(database_url)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
