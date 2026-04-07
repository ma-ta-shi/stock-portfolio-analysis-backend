from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud, schemas
from ..database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/items")
def create_item(item: schemas.ItemCreate, db: Session = Depends(get_db)):
    return crud.create_item(db, item)

@router.get("/items")
def read_items(db: Session = Depends(get_db)):
    return crud.get_items(db)