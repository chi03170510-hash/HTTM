import { useQuery } from '@tanstack/react-query';
import { violationService } from '../services';

export const useViolations = (params: {
  page?: number;
  page_size?: number;
  date_from?: string;
  date_to?: string;
  face_id?: number;
}) => {
  return useQuery({
    // Query key bao gồm cả params để React Query tự động refetch khi params thay đổi (ví dụ: đổi trang)
    queryKey: ['violations', params],
    queryFn: () => violationService.getViolations(params),
    // Giữ dữ liệu cũ hiển thị trên màn hình trong lúc tải trang mới (tránh bị nháy trắng)
    placeholderData: (previousData) => previousData, 
  });
};
