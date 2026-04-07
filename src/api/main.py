from fastapi import FastAPI
from api.database import Base, engine
from api.routes import items

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(items.router)