from fastapi import FastAPI
from api.database import Base, engine
from api.routes import user_profile

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user_profile.router)