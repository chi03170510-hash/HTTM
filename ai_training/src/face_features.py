"""Optional MediaPipe face features and solvePnP head orientation."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

# MediaPipe Face Mesh eye contours, six points each.
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
POSE_IDS = (1, 152, 33, 263, 61, 291)
# 3D template order: nose, chin, subject-left/right eye corners, mouth corners.
# Keep chin below the nose in image coordinates (positive model Y) to avoid the
# 180-degree pitch ambiguity produced by the previous inverted chin point.
MODEL_POINTS = np.array([(0,0,0),(0,63,-12),(-43,32,-26),(43,32,-26),(-28,-28,-24),(28,-28,-24)], dtype=np.float64)

def _dist(a,b): return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))
def eye_aspect_ratio(points):
    p=np.asarray(points,dtype=float)
    if p.shape != (6,2) or not np.isfinite(p).all(): return None
    den=2*_dist(p[0],p[3])
    return None if den < 1e-6 else (_dist(p[1],p[5])+_dist(p[2],p[4]))/den

def head_pose(landmarks, width:int, height:int):
    try:
        import cv2
        image_pts=np.array([(landmarks[i].x*width, landmarks[i].y*height) for i in POSE_IDS], dtype=np.float64)
        focal=float(width); camera=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]],dtype=np.float64)
        ok,rvec,tvec=cv2.solvePnP(MODEL_POINTS,image_pts,camera,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return {"pitch_deg":None,"yaw_deg":None,"roll_deg":None,"valid":False}
        rot,_=cv2.Rodrigues(rvec); angles=cv2.RQDecomp3x3(rot)[0]
        pitch, yaw, roll = map(float, angles)
        # The six-point approximation becomes ambiguous near an inverted face.
        # Mark that observation unavailable instead of publishing a misleading
        # ~180-degree pitch as usable head orientation.
        if not np.isfinite([pitch, yaw, roll]).all() or any(abs(x) > 90.0 for x in (pitch, yaw, roll)):
            return {"pitch_deg":None,"yaw_deg":None,"roll_deg":None,"valid":False}
        return {"pitch_deg":pitch,"yaw_deg":yaw,"roll_deg":roll,"valid":True}
    except Exception: return {"pitch_deg":None,"yaw_deg":None,"roll_deg":None,"valid":False}

def head_pose_from_transform(matrix):
    """Decode MediaPipe FaceLandmarker canonical-to-face transform into Euler angles."""
    try:
        import cv2
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (4, 4) or not np.isfinite(m).all():
            raise ValueError("expected finite 4x4 face transform")
        u, _, vh = np.linalg.svd(m[:3, :3])
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        pitch, yaw, roll = map(float, cv2.RQDecomp3x3(rotation)[0])
        if not np.isfinite([pitch, yaw, roll]).all() or any(abs(x) > 90.0 for x in (pitch, yaw, roll)):
            raise ValueError("implausible head pose")
        return {"pitch_deg":pitch,"yaw_deg":yaw,"roll_deg":roll,"valid":True,"method":"mediapipe_transform"}
    except Exception:
        return {"pitch_deg":None,"yaw_deg":None,"roll_deg":None,"valid":False,"method":"mediapipe_transform"}

class FaceFeatureExtractor:
    def __init__(self, min_detection_confidence=.5, model_path=None):
        self.min_conf = min_detection_confidence
        self.model_path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
        self._mesh = None
        self._backend = None

    @staticmethod
    def availability(model_path=None):
        """Return (usable, reason); Tasks API needs a separately supplied .task asset."""
        try:
            import mediapipe as mp
        except ImportError:
            return False, "mediapipe package is not installed"
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return True, "legacy Face Mesh API"
        path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
        if not (hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")):
            return False, "installed MediaPipe has no supported Face Landmarker API"
        if not path.is_file():
            return False, f"MediaPipe Tasks API requires a Face Landmarker model asset: {path}"
        return True, "MediaPipe Tasks FaceLandmarker"

    def _get_mesh(self):
        if self._mesh is None:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                    refine_landmarks=True, min_detection_confidence=self.min_conf)
                self._backend = "solutions"
            else:
                available, reason = self.availability(self.model_path)
                if not available:
                    raise FileNotFoundError(reason)
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=self.min_conf,
                    min_face_presence_confidence=self.min_conf,
                    output_facial_transformation_matrixes=True,
                )
                self._mesh = mp.tasks.vision.FaceLandmarker.create_from_options(options)
                self._backend = "tasks"
        return self._mesh

    def extract(self, image_bgr):
        import mediapipe as mp
        import cv2
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mesh = self._get_mesh()
        if self._backend == "solutions":
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                return {"ear_left":None,"ear_right":None,"eyes_valid":False,"mouth_open_ratio":None,"mouth_center_xy":None,"head_pose":head_pose([],w,h)}
            lm = result.multi_face_landmarks[0].landmark
        else:
            result = mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.face_landmarks:
                return {"ear_left":None,"ear_right":None,"eyes_valid":False,"mouth_open_ratio":None,"mouth_center_xy":None,"head_pose":head_pose([],w,h)}
            lm = result.face_landmarks[0]
        xy=lambda ids: [(lm[i].x*w,lm[i].y*h) for i in ids]
        left,right=eye_aspect_ratio(xy(LEFT_EYE)),eye_aspect_ratio(xy(RIGHT_EYE))
        pose = head_pose(lm,w,h)
        mouth_width = _dist(xy((61,))[0],xy((291,))[0])
        mouth_open_ratio = _dist(xy((13,))[0],xy((14,))[0])/mouth_width if mouth_width > 1e-6 else None
        mouth_center_xy = [float((lm[13].x+lm[14].x)*.5*w), float((lm[13].y+lm[14].y)*.5*h)]
        matrices = getattr(result, "facial_transformation_matrixes", None) if self._backend == "tasks" else None
        if matrices:
            pose = head_pose_from_transform(matrices[0])
        return {"ear_left":left,"ear_right":right,"eyes_valid":left is not None and right is not None,
                "mouth_open_ratio":mouth_open_ratio,"mouth_center_xy":mouth_center_xy,"head_pose":pose}

    def close(self):
        if self._mesh is not None and hasattr(self._mesh, "close"):
            self._mesh.close()
        self._mesh = None
