"use client";

import React, { useEffect, useRef, useMemo } from 'react';

interface HeatmapProps {
  data?: { topic: string; platform: string; value: number }[];
}

const PLATFORMS = ['Reddit', 'HackerNews', 'GDELT', 'RSS', 'YouTube'];
const TOPICS = ['AI Safety', 'Climate', 'Crypto', 'Elections', 'Open Source', 'Cybersecurity', 'Space'];

function generateMockData() {
  const data: { topic: string; platform: string; value: number }[] = [];
  for (const topic of TOPICS) {
    for (const platform of PLATFORMS) {
      data.push({ topic, platform, value: Math.random() });
    }
  }
  return data;
}

export default function VitalityHeatmap({ data }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapData = useMemo(() => data || generateMockData(), [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellW = canvas.width / PLATFORMS.length;
    const cellH = (canvas.height - 30) / TOPICS.length;
    const offsetY = 20;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw header
    ctx.fillStyle = '#888';
    ctx.font = '10px Inter, sans-serif';
    PLATFORMS.forEach((p, i) => {
      ctx.fillText(p, i * cellW + cellW / 2 - 20, 14);
    });

    // Draw cells
    heatmapData.forEach(({ topic, platform, value }) => {
      const col = PLATFORMS.indexOf(platform);
      const row = TOPICS.indexOf(topic);
      if (col < 0 || row < 0) return;

      const r = Math.floor(255 * value);
      const g = Math.floor(80 * (1 - value));
      const b = Math.floor(200 * (1 - value));
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;

      const x = col * cellW + 1;
      const y = row * cellH + offsetY + 1;
      ctx.beginPath();
      ctx.roundRect(x, y, cellW - 2, cellH - 2, 4);
      ctx.fill();

      // Value text
      ctx.fillStyle = value > 0.5 ? '#fff' : '#ccc';
      ctx.font = '11px monospace';
      ctx.fillText(value.toFixed(2), x + cellW / 2 - 14, y + cellH / 2 + 4);
    });

    // Y-axis labels
    ctx.fillStyle = '#888';
    ctx.font = '10px Inter, sans-serif';
    TOPICS.forEach((t, i) => {
      ctx.save();
      ctx.translate(0, i * cellH + offsetY + cellH / 2 + 4);
      ctx.fillText(t, -70, 0);
      ctx.restore();
    });
  }, [heatmapData]);

  return (
    <div className="w-full h-full flex items-center justify-center overflow-hidden">
      <canvas
        ref={canvasRef}
        width={500}
        height={350}
        style={{ marginLeft: '80px' }}
      />
    </div>
  );
}
