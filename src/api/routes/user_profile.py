from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.crud import user_profile
from api.schemas.user_profile import UserProfileCreate, UserProfileResponse
from api.database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=UserProfileResponse)
def create(user: UserProfileCreate, db: Session = Depends(get_db)):
    return user_profile.create_user_profile(db, user)

@router.get("/", response_model=list[UserProfileResponse])
def read_all(db: Session = Depends(get_db)):
    return user_profile.get_users(db)