import { create } from 'zustand';

interface AppState {
  isCameraActive: boolean;
  setCameraActive: (active: boolean) => void;
  currentSessionId: string | null;
  setCurrentSessionId: (id: string | null) => void;
  selectedCamera: string;
  setSelectedCamera: (cam: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  isCameraActive: false,
  setCameraActive: (active) => set({ isCameraActive: active }),
  currentSessionId: null,
  setCurrentSessionId: (id) => set({ currentSessionId: id }),
  selectedCamera: 'Cam 01 (Bàn 1-3)',
  setSelectedCamera: (cam) => set({ selectedCamera: cam }),
}));
