from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    face_id = Column(Integer, nullable=False, index=True)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
