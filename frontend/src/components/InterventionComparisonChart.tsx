"use client";

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface Step { t: number; I: number; Z: number; }

/**
 * Overlays the baseline vs. treatment trajectories for a chosen compartment
 * so the causal effect of the intervention is visible at a glance.
 */
export default function InterventionComparisonChart({
  baseline,
  treatment,
  metric = 'I',
}: {
  baseline: Step[];
  treatment: Step[];
  metric?: 'I' | 'Z';
}) {
  if (!baseline?.length || !treatment?.length) {
    return <div className="text-gray-400">Run a comparison to see results.</div>;
  }

  const merged = baseline.map((b, i) => ({
    t: b.t,
    baseline: (b as any)[metric],
    treatment: (treatment[i] as any)?.[metric] ?? null,
  }));

  const label = metric === 'I' ? 'Infected (actively sharing)' : 'Zombie (algorithmic amplification)';

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={merged} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="t" stroke="#888" />
          <YAxis stroke="#888" />
          <Tooltip contentStyle={{ backgroundColor: '#111', borderColor: '#333' }} />
          <Legend />
          <Line type="monotone" dataKey="baseline" stroke="#d0021b" dot={false} strokeWidth={2} name={`Baseline ${label}`} />
          <Line type="monotone" dataKey="treatment" stroke="#7ed321" dot={false} strokeWidth={2} name={`With Intervention`} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
