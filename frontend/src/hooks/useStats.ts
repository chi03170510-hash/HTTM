import { useQuery } from '@tanstack/react-query';
import { statsService } from '../services';

export const useStats = () => {
  return useQuery({
    queryKey: ['stats'],
    queryFn: statsService.getStats,
    // Tự động refetch sau mỗi 30 giây để cập nhật số liệu mới nhất
    refetchInterval: 30000, 
  });
};
