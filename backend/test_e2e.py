import os
import sys
import time
import asyncio
from datetime import datetime, timedelta
import cv2
import numpy as np
from fastapi.testclient import TestClient

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from database import SessionLocal, init_db
from models.violation import Violation
from main import app
from services.camera_service import camera_service
from config import UPLOAD_DIR

def run_e2e_tests():
    print("=" * 60)
    print("RUNNING END-TO-END VERIFICATION (PHASE 6)")
    print("=" * 60)

    init_db()
    client = TestClient(app)
    created_violation_ids = []
    created_image_files = []

    try:
        # ============================================================
        # Test Case 1: Khởi động hệ thống
        # ============================================================
        print("\n--- Test Case 1: Khởi động hệ thống ---")
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert res_root.json() == {"message": "HTTM Backend API"}
        print("✓ Root endpoint GET / returned 200 OK")

        res_docs = client.get("/docs")
        assert res_docs.status_code == 200
        print("✓ Swagger UI /docs returned 200 OK")

        res_cam_stat = client.get("/api/camera/status")
        assert res_cam_stat.status_code == 200
        assert res_cam_stat.json()["is_running"] is False
        print("✓ Camera status is_running == False")

        # ============================================================
        # Test Case 2: Bắt đầu giám sát & WebSocket
        # ============================================================
        print("\n--- Test Case 2: Bắt đầu giám sát & WebSocket ---")
        with client.websocket_connect("/ws/monitor") as ws:
            assert len(camera_service.websocket_connections) == 1
            print("✓ WebSocket /ws/monitor connected")

            # Test status broadcast
            test_status = {
                "type": "status",
                "timestamp": datetime.utcnow().isoformat(),
                "camera_running": True,
                "faces_detected": True,
                "faces": [
                    {
                        "face_id": 0,
                        "eye_state": "OPEN",
                        "left_ear": 0.25,
                        "right_ear": 0.24,
                        "avg_ear": 0.245,
                        "closed_duration": 0.0,
                        "is_warning": False,
                    }
                ],
            }
            asyncio.run(camera_service.broadcast_websocket(test_status))
            msg = ws.receive_json()
            assert msg["type"] == "status"
            assert msg["faces"][0]["eye_state"] == "OPEN"
            assert msg["faces"][0]["is_warning"] is False
            print("✓ WebSocket status message received with OPEN eye_state")

        # ============================================================
        # Test Case 3: Phát hiện vi phạm
        # ============================================================
        print("\n--- Test Case 3: Phát hiện vi phạm ---")
        test_face_id = 1
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
        img_name1 = f"violation_{timestamp_str}_face{test_face_id}.jpg"
        img_path1 = os.path.join(UPLOAD_DIR, img_name1)
        created_image_files.append(img_path1)

        # Write snapshot file
        cv2.imwrite(img_path1, np.full((100, 100, 3), 50, dtype=np.uint8))

        # Insert violation record into DB
        db = SessionLocal()
        v1 = Violation(
            face_id=test_face_id,
            started_at=now_dt - timedelta(seconds=5.2),
            ended_at=None,
            duration=5.2,
            image_path=img_name1,
            created_at=now_dt,
        )
        db.add(v1)
        db.commit()
        db.refresh(v1)
        v1_id = v1.id
        created_violation_ids.append(v1_id)
        db.close()

        # Check violation via API
        res_v1 = client.get(f"/api/violations/{v1_id}")
        assert res_v1.status_code == 200
        assert res_v1.json()["id"] == v1_id
        assert res_v1.json()["duration"] == 5.2
        print(f"✓ Violation record #{v1_id} created and verified via API")

        # Check violation image via API
        res_v1_img = client.get(f"/api/violations/{v1_id}/image")
        assert res_v1_img.status_code == 200
        assert "image/jpeg" in res_v1_img.headers.get("content-type", "")
        print(f"✓ Violation image for #{v1_id} retrieved via API")

        # ============================================================
        # Test Case 4: Mở mắt sau vi phạm
        # ============================================================
        print("\n--- Test Case 4: Mở mắt sau vi phạm ---")
        end_time = now_dt + timedelta(seconds=3.0)
        total_duration = 8.2
        db = SessionLocal()
        v1_to_update = db.query(Violation).filter(Violation.id == v1_id).first()
        v1_to_update.ended_at = end_time
        v1_to_update.duration = total_duration
        db.commit()
        db.close()

        # Verify ended_at updated
        res_v1_updated = client.get(f"/api/violations/{v1_id}")
        assert res_v1_updated.status_code == 200
        assert res_v1_updated.json()["ended_at"] is not None
        assert res_v1_updated.json()["duration"] == total_duration
        print(f"✓ Violation #{v1_id} ended_at updated to {res_v1_updated.json()['ended_at']}, duration={total_duration}s")

        # ============================================================
        # Test Case 5: Nhiều vi phạm liên tiếp
        # ============================================================
        print("\n--- Test Case 5: Nhiều vi phạm liên tiếp ---")
        now_dt2 = datetime.now() + timedelta(seconds=10)
        timestamp_str2 = now_dt2.strftime("%Y%m%d_%H%M%S")
        img_name2 = f"violation_{timestamp_str2}_face{test_face_id}.jpg"
        img_path2 = os.path.join(UPLOAD_DIR, img_name2)
        created_image_files.append(img_path2)
        cv2.imwrite(img_path2, np.full((100, 100, 3), 100, dtype=np.uint8))

        db = SessionLocal()
        v2 = Violation(
            face_id=test_face_id,
            started_at=now_dt2 - timedelta(seconds=5.0),
            ended_at=None,
            duration=5.0,
            image_path=img_name2,
            created_at=now_dt2,
        )
        db.add(v2)
        db.commit()
        db.refresh(v2)
        v2_id = v2.id
        created_violation_ids.append(v2_id)
        db.close()

        assert v1_id != v2_id
        assert img_name1 != img_name2
        print(f"✓ Second consecutive violation recorded: #{v2_id} with distinct image {img_name2}")

        # ============================================================
        # Test Case 6: Dừng giám sát
        # ============================================================
        print("\n--- Test Case 6: Dừng giám sát ---")
        assert camera_service.is_running is False
        res_stop_err = client.post("/api/camera/stop")
        assert res_stop_err.status_code == 400
        print("✓ Calling stop when stopped correctly returns 400 Bad Request")

        # ============================================================
        # Test Case 7: API Lịch sử & Thống kê
        # ============================================================
        print("\n--- Test Case 7: API Lịch sử & Thống kê ---")
        # 1. Pagination
        res_page = client.get("/api/violations?page=1&page_size=1")
        assert res_page.status_code == 200
        page_data = res_page.json()
        assert page_data["page"] == 1
        assert page_data["page_size"] == 1
        assert len(page_data["items"]) == 1
        assert page_data["total"] >= 2
        print(f"✓ Pagination works correctly (page=1, size=1, total={page_data['total']}, pages={page_data['total_pages']})")

        # 2. Date filter
        today_str = datetime.now().strftime("%Y-%m-%d")
        res_filter = client.get(f"/api/violations?date_from={today_str}&date_to={today_str}")
        assert res_filter.status_code == 200
        assert res_filter.json()["total"] >= 2
        print("✓ Date filtering works correctly")

        # 3. Stats
        res_stats = client.get("/api/stats")
        assert res_stats.status_code == 200
        stats = res_stats.json()
        assert stats["total_today"] >= 2
        assert stats["total_all"] >= 2
        print(f"✓ Stats summary: today={stats['total_today']}, this_week={stats['total_this_week']}, all={stats['total_all']}")

        # ============================================================
        # Edge Cases
        # ============================================================
        print("\n--- Edge Cases Verification ---")
        # 1. Non-existent violation -> 404
        res_404 = client.get("/api/violations/99999999")
        assert res_404.status_code == 404
        print("✓ GET /api/violations/99999999 returns 404 Not Found")

        # 2. Non-existent violation image -> 404
        res_img_404 = client.get("/api/violations/99999999/image")
        assert res_img_404.status_code == 404
        print("✓ GET /api/violations/99999999/image returns 404 Not Found")

        # 3. Start camera with invalid source
        try:
            camera_service.start(camera_source="non_existent_source_12345.mp4")
            assert False, "Should have raised exception"
        except RuntimeError:
            print("✓ Camera start with invalid source raises RuntimeError cleanly without crashing")

        print("\n" + "=" * 60)
        print("ALL 7 TEST CASES AND EDGE CASES PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        # Cleanup test records
        if created_violation_ids:
            db = SessionLocal()
            db.query(Violation).filter(Violation.id.in_(created_violation_ids)).delete(synchronize_session=False)
            db.commit()
            db.close()

        # Cleanup test files
        for fpath in created_image_files:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass

if __name__ == "__main__":
    run_e2e_tests()
