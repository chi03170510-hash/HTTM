import os
from datetime import datetime, date, time as dt_time, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models.violation import Violation
from schemas.violation import (
    ViolationResponse,
    ViolationListResponse,
    StatsResponse,
)
from config import UPLOAD_DIR

router = APIRouter()


@router.get("/violations", response_model=ViolationListResponse)
def get_violations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    date_from: Optional[date] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    face_id: Optional[int] = Query(None, description="Filter by face_id"),
    db: Session = Depends(get_db),
):
    """
    Returns a paginated list of violations with optional filters.
    """
    query = db.query(Violation)

    if date_from is not None:
        start_datetime = datetime.combine(date_from, dt_time.min)
        query = query.filter(Violation.created_at >= start_datetime)

    if date_to is not None:
        end_datetime = datetime.combine(date_to, dt_time.max)
        query = query.filter(Violation.created_at <= end_datetime)

    if face_id is not None:
        query = query.filter(Violation.face_id == face_id)

    total = query.count()
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    items = (
        query.order_by(Violation.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return ViolationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/stats", response_model=StatsResponse)
def get_violation_stats(db: Session = Depends(get_db)):
    """
    Returns violation counts for today, this week, this month, and all time.
    """
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)
    start_of_week = today_start - timedelta(days=now.weekday())
    start_of_month = datetime(now.year, now.month, 1)

    total_today = db.query(Violation).filter(Violation.created_at >= today_start).count()
    total_this_week = db.query(Violation).filter(Violation.created_at >= start_of_week).count()
    total_this_month = db.query(Violation).filter(Violation.created_at >= start_of_month).count()
    total_all = db.query(Violation).count()
    duration_stats = db.query(
        func.min(Violation.duration),
        func.max(Violation.duration),
        func.avg(Violation.duration),
    ).one()

    return StatsResponse(
        total_today=total_today,
        total_this_week=total_this_week,
        total_this_month=total_this_month,
        total_all=total_all,
        min_duration=float(duration_stats[0]) if duration_stats[0] is not None else None,
        max_duration=float(duration_stats[1]) if duration_stats[1] is not None else None,
        avg_duration=float(duration_stats[2]) if duration_stats[2] is not None else None,
    )


@router.get("/violations/{id}", response_model=ViolationResponse)
def get_violation(id: int, db: Session = Depends(get_db)):
    """
    Returns a single violation by ID.
    """
    violation = db.query(Violation).filter(Violation.id == id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Không tìm thấy vi phạm")
    return violation


@router.get("/violations/{id}/image")
def get_violation_image(id: int, db: Session = Depends(get_db)):
    """
    Returns the snapshot image for a violation.
    """
    violation = db.query(Violation).filter(Violation.id == id).first()
    if not violation:
        raise HTTPException(status_code=404, detail="Không tìm thấy vi phạm")

    if not violation.image_path:
        raise HTTPException(status_code=404, detail="Vi phạm không có thông tin ảnh")

    filename = os.path.basename(violation.image_path)
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File ảnh không tồn tại trên hệ thống")

    return FileResponse(file_path, media_type="image/jpeg")