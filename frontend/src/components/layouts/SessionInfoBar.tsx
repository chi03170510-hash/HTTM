import { Calendar, History, SquareArrowOutUpRight } from 'lucide-react';

interface SessionInfoBarProps {
    room: string;
    shift: string;
    timeRange: string;
    date: string;
    activeTab: 'current' | 'history';
    onTabChange: (tab: 'current' | 'history') => void;
}

const SessionInfoBar = ({
    room = 'PHÒNG A2-302',
    shift = 'Ca sáng',
    timeRange = '08:30 - 11:00',
    date = '15/10/2025',
    activeTab = 'current',
    onTabChange,
}: SessionInfoBarProps) => {
    return (
        <div className="w-full bg-white rounded-xl border border-gray-200 p-2 px-4 flex flex-col sm:flex-row justify-between items-center shadow-sm gap-4 sm:gap-0">

            {/* 1. Phần thông tin buổi học (Bên trái) */}
            <div className="flex items-center gap-4 w-full sm:w-auto">
                {/* Badge Phòng */}
                <div className="flex items-center gap-2 bg-red-50 text-red-600 px-3 py-1.5 rounded-full text-xs font-bold tracking-wide">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                    {room}
                </div>

                {/* Thông tin Ca và Ngày */}
                <div className="flex items-center gap-2 text-sm text-gray-600 font-medium">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    <span>
                        {shift}: {timeRange} <span className="text-gray-300 mx-1">·</span> {date}
                    </span>
                </div>
            </div>

            {/* 2. Phần chuyển đổi Tab (Bên phải) */}
            <div className="flex items-center bg-blue-50 rounded-full p-1 w-full sm:w-auto justify-center">
                {/* Tab Buổi hiện tại */}
                <button
                    onClick={() => onTabChange('current')}
                    className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 ${activeTab === 'current'
                            ? 'bg-white text-blue-600 shadow-sm border border-blue-100'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                >
                    <SquareArrowOutUpRight className="w-3.5 h-3.5" />
                    Buổi hiện tại
                </button>

                {/* Tab Lịch sử tuần */}
                <button
                    onClick={() => onTabChange('history')}
                    className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 ${activeTab === 'history'
                            ? 'bg-white text-blue-600 shadow-sm border border-blue-100'
                            : 'text-gray-500 hover:text-gray-700'
                        }`}
                >
                    <History className="w-3.5 h-3.5" />
                    Lịch sử tuần
                </button>
            </div>
        </div>
    );
};

export default SessionInfoBar;