import React, { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../services/api';
import confetti from 'canvas-confetti';

export interface ProgressState {
  completedModules: number[];
  practicalRuns: Record<number, number>; // moduleId -> count
  quizScores: Record<number, number>; // moduleId -> score out of 100
  notes: Record<number, string>;
}

interface LabContextType {
  activeModule: number;
  setActiveModule: (id: number) => void;
  progress: ProgressState;
  markModuleComplete: (moduleId: number) => void;
  recordPracticalRun: (moduleId: number) => void;
  recordQuizScore: (moduleId: number, score: number) => void;
  resetProgress: () => void;
  serverStatus: {
    online: boolean;
    pytorch_version: string;
    device: string;
  };
  checkServerHealth: () => void;
}

const STORAGE_KEY = 'dl_lab_learning_progress_v1';

const defaultProgress: ProgressState = {
  completedModules: [],
  practicalRuns: {},
  quizScores: {},
  notes: {},
};

const LabContext = createContext<LabContextType | undefined>(undefined);

export const LabProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [activeModule, setActiveModule] = useState<number>(() => {
    const saved = localStorage.getItem('dl_lab_active_module');
    return saved ? parseInt(saved, 10) : 0; // 0 = Dashboard
  });

  const [progress, setProgress] = useState<ProgressState>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : defaultProgress;
    } catch {
      return defaultProgress;
    }
  });

  const [serverStatus, setServerStatus] = useState({
    online: false,
    pytorch_version: 'Connecting...',
    device: 'Unknown',
  });

  useEffect(() => {
    localStorage.setItem('dl_lab_active_module', activeModule.toString());
  }, [activeModule]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
    } catch (e) {
      console.error('Failed to save progress to localStorage', e);
    }
  }, [progress]);

  const checkServerHealth = async () => {
    try {
      const data = await api.getHealth();
      setServerStatus({
        online: true,
        pytorch_version: data.pytorch_version,
        device: data.device.toUpperCase(),
      });
    } catch {
      setServerStatus({
        online: false,
        pytorch_version: 'Offline',
        device: 'None',
      });
    }
  };

  useEffect(() => {
    checkServerHealth();
    const interval = setInterval(checkServerHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const markModuleComplete = (moduleId: number) => {
    setProgress((prev) => {
      if (prev.completedModules.includes(moduleId)) return prev;
      // Trigger celebration confetti
      try {
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#6366f1', '#06b6d4', '#10b981', '#f59e0b'],
        });
      } catch {
        // ignore
      }
      return {
        ...prev,
        completedModules: [...prev.completedModules, moduleId],
      };
    });
  };

  const recordPracticalRun = (moduleId: number) => {
    setProgress((prev) => {
      const runs = prev.practicalRuns[moduleId] || 0;
      const updated = {
        ...prev,
        practicalRuns: { ...prev.practicalRuns, [moduleId]: runs + 1 },
      };
      if (!prev.completedModules.includes(moduleId)) {
        updated.completedModules = [...prev.completedModules, moduleId];
      }
      return updated;
    });
  };

  const recordQuizScore = (moduleId: number, score: number) => {
    setProgress((prev) => ({
      ...prev,
      quizScores: { ...prev.quizScores, [moduleId]: score },
    }));
  };

  const resetProgress = () => {
    if (window.confirm('Are you sure you want to reset all learning progress?')) {
      setProgress(defaultProgress);
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  return (
    <LabContext.Provider
      value={{
        activeModule,
        setActiveModule,
        progress,
        markModuleComplete,
        recordPracticalRun,
        recordQuizScore,
        resetProgress,
        serverStatus,
        checkServerHealth,
      }}
    >
      {children}
    </LabContext.Provider>
  );
};

export const useLab = () => {
  const context = useContext(LabContext);
  if (!context) throw new Error('useLab must be used within a LabProvider');
  return context;
};
