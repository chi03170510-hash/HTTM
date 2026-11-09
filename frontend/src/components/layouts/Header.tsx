import { NavLink } from 'react-router-dom';
import { Video, Bell, User } from 'lucide-react';

const Header = () => {
    return (
        <header className="w-full bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shadow-sm">
            {/* 1. Phần Logo (Bên trái) */}
            <div className="flex items-center gap-3">
                <div className="bg-blue-600 p-2 rounded-lg flex items-center justify-center">
                    <Video className="w-6 h-6 text-white" />
                </div>
                <h1 className="text-xl font-bold text-gray-900 tracking-tight">CLASSROOM</h1>
            </div>

            {/* 2. Phần Menu Điều Hướng (Ở giữa) */}
            <nav className="bg-blue-50 rounded-full px-1 py-1 flex items-center">
                <NavLink
                    to="/"
                    className={({ isActive }) =>
                        `px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${isActive
                            ? 'bg-white text-blue-600 shadow-sm'
                            : 'text-gray-600 hover:text-gray-900'
                        }`
                    }
                >
                    Trang chủ
                </NavLink>
                <NavLink
                    to="/violations" // Thay đổi đường dẫn này tùy theo router của bạn
                    className={({ isActive }) =>
                        `px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${isActive
                            ? 'bg-white text-blue-600 shadow-sm'
                            : 'text-gray-600 hover:text-gray-900'
                        }`
                    }
                >
                    Nhật ký vi phạm
                </NavLink>
            </nav>

            {/* 3. Phần Thông tin User & Thông báo (Bên phải) */}
            <div className="flex items-center gap-6">
                {/* Nút thông báo */}
                <button className="relative p-1 text-gray-500 hover:text-gray-700 transition-colors">
                    <Bell className="w-6 h-6" />
                    {/* Chấm đỏ thông báo */}
                    <span className="absolute top-0 right-0 block h-2.5 w-2.5 rounded-full bg-red-500 ring-2 ring-white"></span>
                </button>

                {/* Thông tin giảng viên */}
                <div className="flex items-center gap-3 border-l border-gray-200 pl-6">
                    <div className="text-right">
                        <p className="text-sm font-semibold text-gray-800">Giảng viên Giám thị</p>
                        <p className="text-xs text-gray-500">Phòng A2-302</p>
                    </div>

                    {/* Avatar */}
                    <div className="bg-blue-600 p-2 rounded-full cursor-pointer hover:bg-blue-700 transition-colors">
                        <User className="w-5 h-5 text-white" />
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;