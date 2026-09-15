import apiClient from './apiClient';
import type { CameraStatus } from '../types';

export const cameraService = {
  /**
   * Khởi động camera và tiến trình AI giám sát
   */
  start: async (): Promise<CameraStatus> => {
    const response = await apiClient.post<CameraStatus>('/camera/start');
    return response.data;
  },
  
  /**
   * Dừng camera và tiến trình AI
   */
  stop: async (): Promise<CameraStatus> => {
    const response = await apiClient.post<CameraStatus>('/camera/stop');
    return response.data;
  },

  /**
   * Lấy trạng thái hiện tại của camera
   */
  getStatus: async (): Promise<CameraStatus> => {
    const response = await apiClient.get<CameraStatus>('/camera/status');
    return response.data;
  }
};
