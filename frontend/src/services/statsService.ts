import apiClient from './apiClient';
import type { Stats } from '../types';

export const statsService = {
  /**
   * Lấy thống kê số lượng vi phạm
   */
  getStats: async (): Promise<Stats> => {
    const response = await apiClient.get<Stats>('/stats');
    return response.data;
  }
};
