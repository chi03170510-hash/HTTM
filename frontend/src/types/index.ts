export interface Violation {
  id: number;
  face_id: number;
  started_at: string; // ISO 8601 date string
  ended_at: string | null; // ISO 8601 date string
  duration: number;
  image_path: string | null;
  created_at: string; // ISO 8601 date string
}

export interface PaginatedViolations {
  items: Violation[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface Stats {
  total_today: number;
  total_this_week: number;
  total_this_month: number;
  total_all: number;
}

export interface CameraStatus {
  is_running: boolean;
  message: string;
}

export interface WSFaceData {
  face_id: number;
  eye_state: 'OPEN' | 'CLOSED';
  left_ear: number;
  right_ear: number;
  avg_ear: number;
  closed_duration: number;
  is_warning: boolean;
}

export interface WSStatusMessage {
  type: 'status';
  timestamp: string;
  camera_running: boolean;
  faces_detected: boolean;
  faces: WSFaceData[];
}

export interface WSWarningMessage {
  type: 'warning';
  timestamp: string;
  face_id: number;
  duration: number;
  violation_id: number;
  image_url: string;
  message: string;
}

export interface WSWarningEndMessage {
  type: 'warning_end';
  timestamp: string;
  face_id: number;
  total_duration: number;
  violation_id: number;
}

export type WebSocketMessage = WSStatusMessage | WSWarningMessage | WSWarningEndMessage | { type: string; [key: string]: any };
