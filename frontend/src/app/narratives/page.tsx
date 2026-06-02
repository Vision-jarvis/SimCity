"use client";

import React from 'react';
import NarrativeTimeline from '@/components/NarrativeTimeline';
import GeoSentimentMap from '@/components/GeoSentimentMap';
import { BookOpen } from 'lucide-react';

export default function NarrativesPage() {
  return (
    <main className="h-screen bg-[#0a0a0a] text-white p-6 overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <BookOpen className="text-purple-400" />
          <div>
            <h1 className="text-2xl font-bold">Narrative Tracker</h1>
            <p className="text-sm text-gray-500">Cross-platform narrative evolution and geographic sentiment</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* Narrative Timeline */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-gray-800">
              <h2 className="text-sm font-bold text-gray-400 uppercase">Narrative: AI Regulation Debate</h2>
              <p className="text-[10px] text-gray-600 mt-1">Cross-platform story evolution over 3 days</p>
            </div>
            <div className="h-[500px]">
              <NarrativeTimeline />
            </div>
          </div>

          {/* Geographic Sentiment */}
          <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-gray-800">
              <h2 className="text-sm font-bold text-gray-400 uppercase">Geographic Sentiment</h2>
              <p className="text-[10px] text-gray-600 mt-1">Regional ideology and emotion shift mapping</p>
            </div>
            <div className="h-[500px] flex items-center justify-center">
              <GeoSentimentMap />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
