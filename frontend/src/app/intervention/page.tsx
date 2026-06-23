"use client";

import React, { useState } from 'react';
import axios from 'axios';
import InterventionComparisonChart from '@/components/InterventionComparisonChart';
import { ShieldCheck, Plus, X, ArrowDown, ArrowUp } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const INTERVENTION_TYPES = [
  { id: 'fact_check', label: 'Fact Check', desc: 'Reduce transmission + algo boost' },
  { id: 'counter_narrative', label: 'Counter Narrative', desc: 'Reduce transmission, neutralize' },
  { id: 'deplatform_bots', label: 'Deplatform Bots', desc: 'Cut the Hawkes surge' },
  { id: 'rate_limit', label: 'Rate Limit', desc: 'Cap cascade growth' },
  { id: 'influencer_amplify', label: 'Influencer Amplify', desc: 'Adversarial: boost spread' },
];

const METRIC_LABELS: Record<string, string> = {
  peak_I: 'Peak Infected',
  total_reach: 'Total Reach',
  final_Z: 'Persistent Misinfo (Z)',
  final_D: 'Debunked / Archived (D)',
  auc_I: 'Cumulative Engagement',
};

interface InterventionRow { type: string; start_step: number; magnitude: number; }

export default function InterventionPage() {
  const [interventions, setInterventions] = useState<InterventionRow[]>([
    { type: 'fact_check', start_step: 10, magnitude: 0.7 },
  ]);
  const [steps, setSteps] = useState(40);
  const [isRunning, setIsRunning] = useState(false);
  const [report, setReport] = useState<any>(null);
  const [metric, setMetric] = useState<'I' | 'Z'>('I');

  const addIntervention = () =>
    setInterventions([...interventions, { type: 'deplatform_bots', start_step: 5, magnitude: 0.5 }]);
  const removeIntervention = (i: number) =>
    setInterventions(interventions.filter((_, idx) => idx !== i));
  const updateIntervention = (i: number, patch: Partial<InterventionRow>) =>
    setInterventions(interventions.map((iv, idx) => (idx === i ? { ...iv, ...patch } : iv)));

  const runComparison = async () => {
    setIsRunning(true);
    setReport(null);
    try {
      const res = await axios.post(`${API_URL}/simulate/intervention`, {
        scenario: { N: 100000, initial_S: 90000, initial_E: 1000, initial_R: 9000, steps, theta: 2.5 },
        interventions,
        include_history: true,
      });
      setReport(res.data);
    } catch (err) {
      console.error('Intervention comparison failed:', err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <main className="h-screen bg-[#0a0a0a] text-white p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <ShieldCheck className="text-green-400" />
          <div>
            <h1 className="text-2xl font-bold">Intervention Simulator</h1>
            <p className="text-sm text-gray-500">
              Counterfactual "what-if": compare a baseline against a world where you intervene.
            </p>
          </div>
        </div>

        {/* Intervention builder */}
        <div className="space-y-3 mb-4">
          {interventions.map((iv, i) => (
            <div key={i} className="flex flex-wrap items-center gap-3 bg-gray-900/50 border border-gray-800 rounded-xl p-3">
              <select
                value={iv.type}
                onChange={(e) => updateIntervention(i, { type: e.target.value })}
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm"
              >
                {INTERVENTION_TYPES.map((t) => (
                  <option key={t.id} value={t.id}>{t.label}</option>
                ))}
              </select>
              <span className="text-[10px] text-gray-500 flex-1 min-w-[140px]">
                {INTERVENTION_TYPES.find((t) => t.id === iv.type)?.desc}
              </span>
              <label className="text-xs text-gray-400">Start step</label>
              <input
                type="number" min={0} value={iv.start_step}
                onChange={(e) => updateIntervention(i, { start_step: Number(e.target.value) })}
                className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
              />
              <label className="text-xs text-gray-400">Magnitude</label>
              <input
                type="range" min={0} max={1} step={0.05} value={iv.magnitude}
                onChange={(e) => updateIntervention(i, { magnitude: Number(e.target.value) })}
              />
              <span className="text-xs text-gray-300 w-8">{iv.magnitude.toFixed(2)}</span>
              <button onClick={() => removeIntervention(i)} className="text-gray-500 hover:text-red-400">
                <X size={16} />
              </button>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-4 mb-6">
          <button onClick={addIntervention} className="flex items-center gap-1 text-xs text-gray-400 hover:text-white">
            <Plus size={14} /> Add intervention
          </button>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400">Steps:</label>
            <input
              type="number" value={steps} onChange={(e) => setSteps(Number(e.target.value))}
              className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs"
            />
          </div>
          <button
            onClick={runComparison}
            disabled={isRunning || interventions.length === 0}
            className="px-6 py-2.5 bg-white text-black font-bold rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
          >
            {isRunning ? 'Computing...' : 'Run Counterfactual'}
          </button>
        </div>

        {/* Results */}
        {report && (
          <div className="space-y-4">
            {/* Delta cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {Object.keys(METRIC_LABELS).map((key) => {
                const pct = report.pct_change?.[key] ?? 0;
                const improved = key === 'final_D' ? pct > 0 : pct < 0; // more debunked = good; less of the rest = good
                return (
                  <div key={key} className="bg-gray-900/50 border border-gray-800 rounded-xl p-3">
                    <div className="text-[10px] text-gray-500 mb-1">{METRIC_LABELS[key]}</div>
                    <div className="text-sm font-bold">
                      {Number(report.baseline_metrics?.[key]).toFixed(0)}
                      <span className="text-gray-600"> → </span>
                      {Number(report.treatment_metrics?.[key]).toFixed(0)}
                    </div>
                    <div className={`text-[11px] flex items-center gap-0.5 mt-1 ${improved ? 'text-green-400' : 'text-red-400'}`}>
                      {pct < 0 ? <ArrowDown size={11} /> : <ArrowUp size={11} />}
                      {Math.abs(pct).toFixed(1)}%
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Overlay chart */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-gray-400">BASELINE vs. INTERVENTION</h2>
                <div className="flex gap-2">
                  {(['I', 'Z'] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setMetric(m)}
                      className={`text-xs px-2 py-1 rounded ${metric === m ? 'bg-gray-700' : 'bg-gray-900 text-gray-500'}`}
                    >
                      {m === 'I' ? 'Infected' : 'Zombie'}
                    </button>
                  ))}
                </div>
              </div>
              <InterventionComparisonChart
                baseline={report.baseline_history}
                treatment={report.treatment_history}
                metric={metric}
              />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
