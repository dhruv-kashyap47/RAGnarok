from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    # bcrypt only uses the first 72 bytes anyway; a cap keeps pathological
    # payloads from doing needless work.
    password: str = Field(min_length=6, max_length=1024)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(max_length=1024)
