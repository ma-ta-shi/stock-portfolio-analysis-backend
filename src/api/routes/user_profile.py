from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from api.crud import user_profile
from api.schemas.user_profile import UserProfileCreate, UserProfileUpdate, UserProfileResponse
from api.database import AsyncSessionLocal

router = APIRouter()


async def get_async_db():
    async with AsyncSessionLocal() as db:
        yield db


@router.post("/", response_model=UserProfileResponse)
async def create(user: UserProfileCreate, db: AsyncSession = Depends(get_async_db)):
    return await user_profile.create_user_profile(db, user)


@router.get("/", response_model=list[UserProfileResponse])
async def read_all(db: AsyncSession = Depends(get_async_db)):
    return await user_profile.get_users(db)


@router.get("/{user_id}", response_model=UserProfileResponse)
async def read_one(user_id: UUID, db: AsyncSession = Depends(get_async_db)):
    result = await user_profile.get_user_profile(db, user_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No user profile found for id {user_id}")
    return result


@router.patch("/{user_id}", response_model=UserProfileResponse)
async def update(user_id: UUID, updates: UserProfileUpdate, db: AsyncSession = Depends(get_async_db)):
    result = await user_profile.update_user_profile(db, user_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No user profile found for id {user_id}")
    return result


@router.delete("/{user_id}", status_code=204)
async def delete(user_id: UUID, db: AsyncSession = Depends(get_async_db)):
    result = await user_profile.delete_user(db, user_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No user profile found for id {user_id}")
