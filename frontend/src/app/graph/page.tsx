"use client";

import React from 'react';
import dynamic from 'next/dynamic';
import { useGraphData } from '@/hooks/useGraphData';
import { RefreshCw, Network } from 'lucide-react';

const Graph3D = dynamic(() => import('@/components/Graph3D'), { ssr: false });

export default function GraphPage() {
  const { data, loading, refresh } = useGraphData();

  return (
    <main className="flex flex-col h-screen bg-[#0a0a0a] text-white">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <Network className="text-blue-400" />
          <div>
            <h1 className="text-lg font-bold">Live Knowledge Graph</h1>
            <p className="text-xs text-gray-500">Real-time internet topology visualization</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-xs text-gray-400">
            <span className="text-blue-400 font-mono">{data.stats.total_nodes.toLocaleString()}</span> nodes
            <span className="mx-2">·</span>
            <span className="text-purple-400 font-mono">{data.stats.total_edges.toLocaleString()}</span> edges
          </div>
          <button
            onClick={refresh}
            disabled={loading}
            className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* 3D Graph */}
      <div className="flex-1 relative">
        <Graph3D infectionRate={0} />
      </div>
    </main>
  );
}
