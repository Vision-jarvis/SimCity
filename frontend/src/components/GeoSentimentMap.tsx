"use client";

import React, { useEffect, useRef } from 'react';

interface RegionData {
  name: string;
  lat: number;
  lng: number;
  sentiment: number;
  volume: number;
}

const MOCK_REGIONS: RegionData[] = [
  { name: 'North America', lat: 40, lng: -100, sentiment: -0.2, volume: 4500 },
  { name: 'Europe', lat: 50, lng: 10, sentiment: -0.1, volume: 3200 },
  { name: 'East Asia', lat: 35, lng: 120, sentiment: 0.1, volume: 2800 },
  { name: 'South Asia', lat: 20, lng: 78, sentiment: -0.3, volume: 1500 },
  { name: 'South America', lat: -15, lng: -60, sentiment: 0.05, volume: 900 },
  { name: 'Africa', lat: 5, lng: 20, sentiment: -0.05, volume: 600 },
  { name: 'Oceania', lat: -25, lng: 135, sentiment: 0.15, volume: 400 },
  { name: 'Middle East', lat: 30, lng: 45, sentiment: -0.4, volume: 1100 },
];

function sentimentToColor(s: number): string {
  if (s > 0.2) return '#4ADE80';
  if (s > 0) return '#86EFAC';
  if (s > -0.2) return '#FDE047';
  if (s > -0.4) return '#FB923C';
  return '#EF4444';
}

export default function GeoSentimentMap({ regions }: { regions?: RegionData[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const data = regions || MOCK_REGIONS;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Dark background with grid
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Simple equirectangular projection
    const projectX = (lng: number) => ((lng + 180) / 360) * w;
    const projectY = (lat: number) => ((90 - lat) / 180) * h;

    // Draw regions as glowing circles
    data.forEach((region) => {
      const x = projectX(region.lng);
      const y = projectY(region.lat);
      const radius = Math.sqrt(region.volume) / 3;
      const color = sentimentToColor(region.sentiment);

      // Glow
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 2);
      gradient.addColorStop(0, color + '66');
      gradient.addColorStop(1, color + '00');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, radius * 2, 0, Math.PI * 2);
      ctx.fill();

      // Core dot
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, radius * 0.5, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = '#ccc';
      ctx.font = '10px Inter, sans-serif';
      ctx.fillText(region.name, x + radius * 0.6 + 4, y + 4);

      // Sentiment badge
      ctx.fillStyle = color;
      ctx.font = '9px monospace';
      ctx.fillText(
        `${region.sentiment > 0 ? '+' : ''}${region.sentiment.toFixed(2)}`,
        x + radius * 0.6 + 4,
        y + 16,
      );
    });
  }, [data]);

  return (
    <div className="w-full h-full relative">
      <canvas ref={canvasRef} width={700} height={400} className="w-full h-full" />
      {/* Legend */}
      <div className="absolute bottom-3 right-3 flex gap-2 text-[10px]">
        {[
          { label: 'Positive', color: '#4ADE80' },
          { label: 'Neutral', color: '#FDE047' },
          { label: 'Negative', color: '#EF4444' },
        ].map((item) => (
          <div key={item.label} className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
            <span className="text-gray-400">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
