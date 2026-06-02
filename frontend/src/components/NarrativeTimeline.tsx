"use client";

import React from 'react';

interface NarrativeEvent {
  id: string;
  timestamp: number;
  platform: string;
  summary: string;
  sentiment: number;
}

const PLATFORM_COLORS: Record<string, string> = {
  reddit: '#FF4500',
  hackernews: '#FF6600',
  gdelt: '#4A90D9',
  rss: '#2ECC71',
  youtube: '#FF0000',
};

const mockEvents: NarrativeEvent[] = [
  { id: '1', timestamp: Date.now() - 86400000 * 3, platform: 'reddit', summary: 'Initial post about AI regulation gains traction on r/technology', sentiment: -0.3 },
  { id: '2', timestamp: Date.now() - 86400000 * 2.5, platform: 'hackernews', summary: 'HN front page: "Why AI regulation matters now"', sentiment: -0.1 },
  { id: '3', timestamp: Date.now() - 86400000 * 2, platform: 'gdelt', summary: 'Major news outlets cover AI regulation proposals', sentiment: -0.2 },
  { id: '4', timestamp: Date.now() - 86400000 * 1.5, platform: 'youtube', summary: 'Tech influencers debate AI safety in viral videos', sentiment: -0.4 },
  { id: '5', timestamp: Date.now() - 86400000, platform: 'reddit', summary: 'Counter-narrative emerges: "AI regulation will stifle innovation"', sentiment: 0.2 },
  { id: '6', timestamp: Date.now() - 43200000, platform: 'rss', summary: 'Blog posts synthesize both sides of the debate', sentiment: 0.0 },
  { id: '7', timestamp: Date.now() - 3600000, platform: 'gdelt', summary: 'Government officials respond to public discourse', sentiment: -0.1 },
];

export default function NarrativeTimeline({ events }: { events?: NarrativeEvent[] }) {
  const timelineEvents = events || mockEvents;

  return (
    <div className="w-full h-full overflow-y-auto p-4">
      <div className="relative ml-6">
        {/* Vertical line */}
        <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-gray-700" />

        {timelineEvents.map((event, i) => {
          const color = PLATFORM_COLORS[event.platform] || '#888';
          const date = new Date(event.timestamp);
          const timeStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
          const sentimentColor = event.sentiment > 0 ? '#4ADE80' : event.sentiment < -0.2 ? '#EF4444' : '#A1A1AA';

          return (
            <div key={event.id} className="relative pl-8 pb-6 group">
              {/* Dot */}
              <div
                className="absolute left-[-5px] top-1.5 w-3 h-3 rounded-full border-2 border-gray-900 transition-transform group-hover:scale-150"
                style={{ backgroundColor: color }}
              />

              {/* Content */}
              <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-3 hover:border-gray-600 transition-colors">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded"
                    style={{ backgroundColor: color + '22', color }}
                  >
                    {event.platform}
                  </span>
                  <span className="text-[10px] text-gray-500">{timeStr}</span>
                  <span
                    className="ml-auto text-[10px] font-mono"
                    style={{ color: sentimentColor }}
                  >
                    {event.sentiment > 0 ? '+' : ''}{event.sentiment.toFixed(2)}
                  </span>
                </div>
                <p className="text-sm text-gray-300">{event.summary}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
