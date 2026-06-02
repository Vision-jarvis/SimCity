"use client";

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import VitalityHeatmap from '@/components/VitalityHeatmap';
import { TrendingUp, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TrendItem {
  topic: string;
  score: number;
  platform: string;
  sentiment: string;
  velocity: number;
}

const SENTIMENT_ICONS: Record<string, React.ReactNode> = {
  positive: <ArrowUpRight size={12} className="text-green-400" />,
  negative: <ArrowDownRight size={12} className="text-red-400" />,
  neutral: <Minus size={12} className="text-gray-400" />,
};

export default function TrendsPage() {
  const [trends, setTrends] = useState<TrendItem[]>([]);

  useEffect(() => {
    axios.get(`${API_URL}/trends/current`).then((res) => {
      setTrends(res.data.trends || []);
    }).catch(() => {
      // Fallback mock data
      setTrends([
        { topic: 'AI Regulation', score: 0.95, platform: 'reddit', sentiment: 'negative', velocity: 2.3 },
        { topic: 'Climate Summit', score: 0.87, platform: 'gdelt', sentiment: 'neutral', velocity: 1.8 },
        { topic: 'Crypto Crash', score: 0.82, platform: 'hackernews', sentiment: 'negative', velocity: 3.1 },
        { topic: 'Open Source LLMs', score: 0.78, platform: 'hackernews', sentiment: 'positive', velocity: 1.5 },
        { topic: 'Election Disinfo', score: 0.75, platform: 'reddit', sentiment: 'negative', velocity: 2.7 },
      ]);
    });
  }, []);

  return (
    <main className="h-screen bg-[#0a0a0a] text-white p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <TrendingUp className="text-green-400" />
          <div>
            <h1 className="text-2xl font-bold">Virality Forecasts</h1>
            <p className="text-sm text-gray-500">Real-time trending topics and virality predictions</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* Trending Topics List */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
            <h2 className="text-sm font-bold text-gray-400 uppercase mb-4">Trending Topics</h2>
            <div className="space-y-2">
              {trends.map((trend, i) => (
                <div
                  key={trend.topic}
                  className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  <span className="text-xs text-gray-600 font-mono w-5">{i + 1}</span>
                  <div className="flex-1">
                    <div className="text-sm font-medium">{trend.topic}</div>
                    <div className="text-[10px] text-gray-500">{trend.platform}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {SENTIMENT_ICONS[trend.sentiment]}
                    <div className="text-right">
                      <div className="text-xs font-mono" style={{
                        color: trend.score > 0.85 ? '#EF4444' : trend.score > 0.7 ? '#F59E0B' : '#4ADE80'
                      }}>
                        {(trend.score * 100).toFixed(0)}%
                      </div>
                      <div className="text-[10px] text-gray-500">v={trend.velocity.toFixed(1)}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Heatmap */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
            <h2 className="text-sm font-bold text-gray-400 uppercase mb-4">Platform × Topic Heatmap</h2>
            <VitalityHeatmap />
          </div>
        </div>
      </div>
    </main>
  );
}
