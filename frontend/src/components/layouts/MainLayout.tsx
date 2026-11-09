import { useState } from 'react';
import Header from './Header';
import SessionInfoBar from './SessionInfoBar';
import { Outlet } from 'react-router-dom';

const MainLayout = () => {
    // State quản lý tab đang được chọn
    const [activeTab, setActiveTab] = useState<'current' | 'history'>('current');

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col">
            <Header />

            {/* Thanh thông tin buổi học */}
            <div className="px-6 pt-4">
                <SessionInfoBar
                    activeTab={activeTab}
                    onTabChange={setActiveTab}
                    // Bạn có thể truyền dữ liệu động từ API vào đây
                    room="PHÒNG A2-302"
                    shift="Ca sáng"
                    timeRange="08:30 - 11:00"
                    date="15/10/2025"
                />
            </div>

            <main className="flex-1 p-6">
                <Outlet context={{ activeTab }} />
            </main>
        </div>
    );
};

export default MainLayout;