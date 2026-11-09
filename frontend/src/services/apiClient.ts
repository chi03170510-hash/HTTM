import axios from 'axios';

const apiClient = axios.create({
  // Sử dụng '/api' làm baseURL vì chúng ta đã cấu hình Vite proxy chuyển tiếp các request này đến backend (localhost:8000)
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor cho request (có thể thêm Token xác thực ở đây sau này nếu cần)
apiClient.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Interceptor cho response (xử lý lỗi chung toàn cục)
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Có thể xử lý thông báo lỗi toàn hệ thống ở đây (ví dụ: toast thông báo)
    return Promise.reject(error);
  }
);

export default apiClient;
