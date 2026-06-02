"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { Play, Pause, SkipForward, RotateCcw } from 'lucide-react';

interface SimulationStep {
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

interface SimulationPlayerProps {
  data: SimulationStep[];
  onStepChange?: (step: number) => void;
}

export default function SimulationPlayer({ data, onStepChange }: SimulationPlayerProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);

  useEffect(() => {
    if (!isPlaying || data.length === 0) return;

    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        const next = prev + 1;
        if (next >= data.length) {
          setIsPlaying(false);
          return prev;
        }
        return next;
      });
    }, speed);

    return () => clearInterval(timer);
  }, [isPlaying, speed, data.length]);

  useEffect(() => {
    onStepChange?.(currentStep);
  }, [currentStep, onStepChange]);

  const reset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const stepForward = () => {
    if (currentStep < data.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const current = data[currentStep] || null;
  const progress = data.length > 0 ? (currentStep / (data.length - 1)) * 100 : 0;

  return (
    <div className="w-full bg-gray-900/50 border border-gray-800 rounded-lg p-4">
      {/* Progress bar */}
      <div className="w-full h-1.5 bg-gray-800 rounded-full mb-4 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-red-500 transition-all duration-300 rounded-full"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={reset}
          className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
        >
          <RotateCcw size={14} />
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="p-2 rounded-lg bg-white text-black hover:bg-gray-200 transition-colors"
        >
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button
          onClick={stepForward}
          className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
        >
          <SkipForward size={14} />
        </button>

        <span className="text-xs text-gray-400 ml-2">
          Day {currentStep + 1} / {data.length}
        </span>

        {/* Speed control */}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-[10px] text-gray-500 uppercase">Speed</span>
          {[1000, 500, 200].map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`text-[10px] px-2 py-1 rounded ${
                speed === s ? 'bg-gray-600 text-white' : 'bg-gray-800 text-gray-400'
              }`}
            >
              {s === 1000 ? '1x' : s === 500 ? '2x' : '5x'}
            </button>
          ))}
        </div>
      </div>

      {/* Current state */}
      {current && (
        <div className="grid grid-cols-6 gap-2 text-center">
          {[
            { label: 'S', value: current.S, color: '#4A90E2' },
            { label: 'E', value: current.E, color: '#F5A623' },
            { label: 'I', value: current.I, color: '#D0021B' },
            { label: 'R', value: current.R, color: '#7ED321' },
            { label: 'Z', value: current.Z, color: '#BD10E0' },
            { label: 'D', value: current.D, color: '#8B572A' },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-gray-800/50 rounded p-2">
              <div className="text-[10px] font-bold" style={{ color }}>{label}</div>
              <div className="text-xs font-mono text-gray-300">
                {Math.floor(value).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
