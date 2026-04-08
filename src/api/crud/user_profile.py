from sqlalchemy.orm import Session
from api.tables.user_profile import UserProfile
from api.schemas.user_profile import UserProfileCreate


def create_user_profile(db: Session, user_profile: UserProfileCreate):
    db_user = UserProfile(display_name=user_profile.display_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_users(db: Session):
    return db.query(UserProfile).all()


def get_user_profile(db: Session, user_id):
    return db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

def delete_user(db: Session, user_id):
    user = get_user_profile(db, user_id)
    if user:
        db.delete(user)
        db.commit()
    return user