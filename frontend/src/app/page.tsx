"use client";

import React, { useState, useEffect } from 'react';
import Graph3D from '@/components/Graph3D';
import SimulateChart from '@/components/SimulateChart';
import axios from 'axios';
import { Play, Activity, Users, ShieldAlert } from 'lucide-react';

export default function Dashboard() {
  const [isRunning, setIsRunning] = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [infectionRate, setInfectionRate] = useState(0);

  // Auto-play the simulation timeline
  useEffect(() => {
    if (history.length > 0 && currentStep < history.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
        
        // Map beta to a visual infection probability for the 3D graph
        const currentData = history[currentStep];
        const newInfections = currentData.I / (currentData.S + currentData.I + 1);
        setInfectionRate(newInfections * 5); // Scale for visual effect
      }, 500); // 500ms per day
      return () => clearTimeout(timer);
    }
  }, [history, currentStep]);

  const runSimulation = async () => {
    setIsRunning(true);
    setHistory([]);
    setCurrentStep(0);
    setInfectionRate(0);

    try {
      const config = {
        scenario_id: "misinfo_outbreak",
        N: 100000,
        initial_S: 90000,
        initial_E: 1000,
        initial_I: 0,
        initial_R: 9000,
        initial_Z: 0,
        initial_D: 0,
        theta: 2.5,
        sigma: 0.7,
        gamma_I: 0.05,
        delta_D: 0.005,
        base_beta_macro: 0.9,
        baseline_lambda: 1.0,
        injected_lambda: 51.0, // Big Hawkes spike!
        decay_gamma: 0.1,
        phi: 0.08,
        steps: 30,
        dt: 1.0
      };

      // Ensure your FastAPI server is running on port 8000
      const res = await axios.post('http://localhost:8000/simulate/run', config);
      setHistory(res.data.results);
    } catch (err) {
      console.error("Simulation failed:", err);
      alert("Failed to connect to simulation engine. Is the Python FastAPI server running?");
    } finally {
      setIsRunning(false);
    }
  };

  const currentStats = history.length > 0 ? history[currentStep] : null;

  return (
    <main className="flex h-screen w-full bg-[#0a0a0a] text-white overflow-hidden p-4 gap-4">
      {/* LEFT COLUMN: 3D GRAPH */}
      <section className="flex-1 rounded-xl overflow-hidden border border-gray-800 shadow-2xl relative">
        <Graph3D infectionRate={infectionRate} />
        
        {/* Overlay Stats */}
        <div className="absolute bottom-6 left-6 flex gap-4">
          <div className="glass-panel p-4 flex items-center gap-3">
            <Users className="text-blue-400" />
            <div>
              <p className="text-xs text-gray-400 uppercase font-bold tracking-wider">Total Population</p>
              <p className="text-xl font-mono">100,000</p>
            </div>
          </div>
          <div className="glass-panel p-4 flex items-center gap-3">
            <ShieldAlert className="text-purple-500" />
            <div>
              <p className="text-xs text-gray-400 uppercase font-bold tracking-wider">Algorithmic Zombies</p>
              <p className="text-xl font-mono text-purple-400">
                {currentStats ? Math.floor(currentStats.Z).toLocaleString() : '0'}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* RIGHT COLUMN: CONTROLS & CHARTS */}
      <aside className="w-96 flex flex-col gap-4">
        {/* Header */}
        <div className="glass-panel p-6">
          <h1 className="text-2xl font-bold tracking-tight mb-1">SimCity Engine</h1>
          <p className="text-sm text-gray-400 mb-6">AI Digital Twin of the Internet</p>
          
          <button 
            onClick={runSimulation}
            disabled={isRunning}
            className="w-full py-3 px-4 bg-white text-black hover:bg-gray-200 font-bold rounded-lg transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {isRunning ? (
              <Activity className="animate-pulse" />
            ) : (
              <Play size={18} />
            )}
            {isRunning ? "Computing..." : "Run Scenario: Misinfo Outbreak"}
          </button>
        </div>

        {/* Chart Panel */}
        <div className="glass-panel p-4 flex-1 flex flex-col">
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-4">SEIR-Z-D Trajectory</h2>
          
          {history.length > 0 ? (
            <div className="flex-1">
              <SimulateChart data={history.slice(0, currentStep + 1)} />
              
              <div className="mt-6 grid grid-cols-2 gap-4 text-sm">
                <div className="bg-gray-900 p-3 rounded border border-gray-800">
                  <p className="text-gray-500 mb-1">Day</p>
                  <p className="font-mono text-lg">{currentStep + 1} / {history.length}</p>
                </div>
                <div className="bg-gray-900 p-3 rounded border border-gray-800">
                  <p className="text-gray-500 mb-1">Hawkes Intensity (λ)</p>
                  <p className="font-mono text-lg text-yellow-500">
                    {currentStats?.lambda.toFixed(2)}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-600 text-sm text-center">
              Awaiting simulation start.<br/>Click Run Scenario to begin.
            </div>
          )}
        </div>
      </aside>
    </main>
  );
}
