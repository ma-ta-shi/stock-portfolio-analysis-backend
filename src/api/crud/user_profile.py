from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.tables.user_profile import UserProfile
from api.schemas.user_profile import UserProfileCreate, UserProfileUpdate


def _dump_fields(data, *, exclude_unset: bool) -> dict:
    """model_dump() in its default "python" mode, except for
    time_sensitive_cash_needs: `date` isn't JSON-serializable, and the JSON
    column needs a plain, storable value. The other fields (floats, real
    datetimes for direct column assignment) go through the default mode
    unaffected. Shared by create and update -- both accept the same
    personalization fields (see _UserProfileFields)."""
    dumped = data.model_dump(exclude_unset=exclude_unset)
    if dumped.get("time_sensitive_cash_needs") is not None:
        dumped["time_sensitive_cash_needs"] = [
            need.model_dump(mode="json") for need in data.time_sensitive_cash_needs
        ]
    return dumped


async def create_user_profile(db: AsyncSession, user_profile: UserProfileCreate) -> UserProfile:
    db_user = UserProfile(**_dump_fields(user_profile, exclude_unset=False))
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def get_users(db: AsyncSession) -> list[UserProfile]:
    result = await db.execute(select(UserProfile))
    return list(result.scalars().all())


async def get_user_profile(db: AsyncSession, user_id) -> UserProfile | None:
    result = await db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
    return result.scalar_one_or_none()


async def update_user_profile(db: AsyncSession, user_id, updates: UserProfileUpdate) -> UserProfile | None:
    user = await get_user_profile(db, user_id)
    if user is None:
        return None
    for field, value in _dump_fields(updates, exclude_unset=True).items():
        setattr(user, field, value)
    await db.commit()
    await db.refresh(user)
    return user


async def delete_user(db: AsyncSession, user_id) -> UserProfile | None:
    user = await get_user_profile(db, user_id)
    if user is not None:
        await db.delete(user)
        await db.commit()
    return user
