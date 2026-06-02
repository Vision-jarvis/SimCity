import { create } from 'zustand';

export interface SimulationStep {
  t: number;
  S: number;
  E: number;
  I: number;
  R: number;
  Z: number;
  D: number;
  beta: number;
  zeta: number;
  lambda: number;
  mean_opinion: number;
}

export interface TrendItem {
  topic: string;
  score: number;
  platform: string;
  sentiment: string;
  velocity: number;
}

export interface StreamEvent {
  type: string;
  timestamp: number;
  platform: string;
  topic: string;
  sentiment: string;
  virality_score: number;
  toxicity: number;
  author: string;
}

interface AppState {
  // Simulation
  simulationHistory: SimulationStep[];
  currentStep: number;
  isRunning: boolean;
  setSimulationHistory: (history: SimulationStep[]) => void;
  setCurrentStep: (step: number) => void;
  setIsRunning: (running: boolean) => void;

  // Trends
  trends: TrendItem[];
  setTrends: (trends: TrendItem[]) => void;

  // Stream
  streamEvents: StreamEvent[];
  addStreamEvent: (event: StreamEvent) => void;
  clearStreamEvents: () => void;

  // Navigation
  activePage: string;
  setActivePage: (page: string) => void;

  // Connection
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  // Simulation
  simulationHistory: [],
  currentStep: 0,
  isRunning: false,
  setSimulationHistory: (history) => set({ simulationHistory: history }),
  setCurrentStep: (step) => set({ currentStep: step }),
  setIsRunning: (running) => set({ isRunning: running }),

  // Trends
  trends: [],
  setTrends: (trends) => set({ trends }),

  // Stream
  streamEvents: [],
  addStreamEvent: (event) =>
    set((state) => ({
      streamEvents: [...state.streamEvents.slice(-99), event],
    })),
  clearStreamEvents: () => set({ streamEvents: [] }),

  // Navigation
  activePage: 'dashboard',
  setActivePage: (page) => set({ activePage: page }),

  // Connection
  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),
}));
