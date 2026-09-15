import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { cameraService } from '../services';

// Hook lấy trạng thái camera
export const useCameraStatus = () => {
  return useQuery({
    queryKey: ['cameraStatus'],
    queryFn: cameraService.getStatus,
    // Tự động kiểm tra trạng thái mỗi 5 giây
    refetchInterval: 5000,
  });
};

// Hook bắt đầu camera
export const useStartCamera = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: cameraService.start,
    onSuccess: () => {
      // Yêu cầu React Query fetch lại trạng thái camera ngay lập tức
      queryClient.invalidateQueries({ queryKey: ['cameraStatus'] });
    },
  });
};

// Hook dừng camera
export const useStopCamera = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: cameraService.stop,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cameraStatus'] });
    },
  });
};
