from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class ViolationResponse(BaseModel):
    """Response schema for a single violation."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    face_id: int
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration: float
    image_path: Optional[str] = None
    created_at: datetime


class ViolationListResponse(BaseModel):
    """Paginated response schema for violation list."""
    items: List[ViolationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class StatsResponse(BaseModel):
    """Response schema for violation statistics."""
    total_today: int
    total_this_week: int
    total_this_month: int
    total_all: int
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    avg_duration: Optional[float] = None


class CameraStatusResponse(BaseModel):
    """Response schema for camera status."""
    is_running: bool
    message: str