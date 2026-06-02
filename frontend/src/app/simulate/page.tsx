"use client";

import React, { useState } from 'react';
import axios from 'axios';
import SimulateChart from '@/components/SimulateChart';
import SimulationPlayer from '@/components/SimulationPlayer';
import { Zap, Settings } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const PRESETS = [
  { id: 'misinfo_outbreak', name: 'Misinfo Outbreak', theta: 2.5, sigma: 0.7, injected_lambda: 51, phi: 0.08, color: '#EF4444' },
  { id: 'influencer_tweet', name: 'Influencer Tweet', theta: 2.0, sigma: 0.5, injected_lambda: 50, phi: 0.05, color: '#8B5CF6' },
  { id: 'platform_outage', name: 'Platform Outage', theta: 1.5, sigma: 0.3, injected_lambda: 30, phi: 0.03, color: '#F59E0B' },
];

export default function SimulatePage() {
  const [history, setHistory] = useState<any[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState(PRESETS[0]);
  const [steps, setSteps] = useState(30);

  const runSimulation = async () => {
    setIsRunning(true);
    setHistory([]);
    try {
      const config = {
        scenario_id: selectedPreset.id,
        N: 100000,
        initial_S: 90000,
        initial_E: 1000,
        initial_I: 0,
        initial_R: 9000,
        initial_Z: 0,
        initial_D: 0,
        theta: selectedPreset.theta,
        sigma: selectedPreset.sigma,
        gamma_I: 0.05,
        delta_D: 0.005,
        base_beta_macro: 0.9,
        baseline_lambda: 1.0,
        injected_lambda: selectedPreset.injected_lambda,
        decay_gamma: 0.1,
        phi: selectedPreset.phi,
        steps,
        dt: 1.0,
      };
      const res = await axios.post(`${API_URL}/simulate/run`, config);
      setHistory(res.data.results);
    } catch (err) {
      console.error('Simulation failed:', err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <main className="h-screen bg-[#0a0a0a] text-white p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <Zap className="text-yellow-400" />
          <div>
            <h1 className="text-2xl font-bold">Scenario Simulator</h1>
            <p className="text-sm text-gray-500">Run "what-if" internet scenarios with SEIR-Z-D dynamics</p>
          </div>
        </div>

        {/* Preset Selection */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => setSelectedPreset(preset)}
              className={`p-4 rounded-xl border transition-all text-left ${
                selectedPreset.id === preset.id
                  ? 'border-white/30 bg-gray-800/80'
                  : 'border-gray-800 bg-gray-900/50 hover:border-gray-700'
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: preset.color }} />
                <span className="text-sm font-bold">{preset.name}</span>
              </div>
              <div className="text-[10px] text-gray-500 space-y-0.5">
                <div>θ={preset.theta} σ={preset.sigma}</div>
                <div>λ={preset.injected_lambda} φ={preset.phi}</div>
              </div>
            </button>
          ))}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4 mb-6">
          <div className="flex items-center gap-2">
            <Settings size={14} className="text-gray-500" />
            <label className="text-xs text-gray-400">Steps:</label>
            <input
              type="number"
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
              className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
            />
          </div>
          <button
            onClick={runSimulation}
            disabled={isRunning}
            className="px-6 py-2.5 bg-white text-black font-bold rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
          >
            {isRunning ? 'Computing...' : 'Run Scenario'}
          </button>
        </div>

        {/* Results */}
        {history.length > 0 && (
          <div className="space-y-4">
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
              <h2 className="text-sm font-bold text-gray-400 mb-4">SEIR-Z-D TRAJECTORY</h2>
              <SimulateChart data={history} />
            </div>
            <SimulationPlayer data={history} />
          </div>
        )}
      </div>
    </main>
  );
}
