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
  ResponsiveContainer
} from 'recharts';

export default function SimulateChart({ data }: { data: any[] }) {
  if (!data || data.length === 0) return <div className="text-gray-400">No simulation data.</div>;

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="t" stroke="#888" />
          <YAxis stroke="#888" />
          <Tooltip contentStyle={{ backgroundColor: '#111', borderColor: '#333' }} />
          <Legend />
          <Line type="monotone" dataKey="S" stroke="#4a90e2" dot={false} strokeWidth={2} name="Susceptible" />
          <Line type="monotone" dataKey="E" stroke="#f5a623" dot={false} strokeWidth={2} name="Exposed" />
          <Line type="monotone" dataKey="I" stroke="#d0021b" dot={false} strokeWidth={2} name="Infected" />
          <Line type="monotone" dataKey="R" stroke="#7ed321" dot={false} strokeWidth={2} name="Recovered" />
          <Line type="monotone" dataKey="Z" stroke="#bd10e0" dot={false} strokeWidth={3} name="Zombie (Algo)" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
