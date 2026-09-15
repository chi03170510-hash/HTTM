import apiClient from './apiClient';
import type { PaginatedViolations, Violation } from '../types';

export const violationService = {
  /**
   * Lấy danh sách các vi phạm (có phân trang và bộ lọc)
   */
  getViolations: async (params?: { 
    page?: number; 
    page_size?: number; 
    date_from?: string; // Định dạng YYYY-MM-DD
    date_to?: string;   // Định dạng YYYY-MM-DD
    face_id?: number 
  }): Promise<PaginatedViolations> => {
    const response = await apiClient.get<PaginatedViolations>('/violations', { params });
    return response.data;
  },

  /**
   * Lấy chi tiết một vi phạm theo ID
   */
  getViolation: async (id: number): Promise<Violation> => {
    const response = await apiClient.get<Violation>(`/violations/${id}`);
    return response.data;
  },

  /**
   * Tạo URL của hình ảnh vi phạm (thường dùng thẳng vào src của thẻ <img>)
   */
  getViolationImageUrl: (id: number): string => {
    // Trả về đường dẫn mà Vite proxy sẽ forward lên server backend
    return `/api/violations/${id}/image`;
  },
  
  /**
   * Tải hình ảnh vi phạm dưới dạng Blob (dành cho trường hợp cần tải xuống hoặc xử lý riêng)
   */
  getViolationImageBlob: async (id: number): Promise<Blob> => {
    const response = await apiClient.get(`/violations/${id}/image`, {
      responseType: 'blob'
    });
    return response.data;
  }
};
