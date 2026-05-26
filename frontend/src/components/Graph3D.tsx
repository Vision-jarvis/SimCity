"use client";

import React, { useRef, useEffect, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import * as THREE from 'three';

// Dynamically import react-force-graph-3d to avoid SSR issues
const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), { ssr: false });

export default function Graph3D({ infectionRate = 0 }: { infectionRate: number }) {
  const fgRef = useRef<any>(null);
  const [graphData, setGraphData] = useState<any>({ nodes: [], links: [] });

  useEffect(() => {
    // Generate a mock internet topology (scale-free-ish)
    const N = 300;
    const nodes = Array.from({ length: N }).map((_, id) => ({
      id,
      val: Math.random() * 1.5 + 0.5,
      group: Math.floor(Math.random() * 3), // Community ID
      status: 'susceptible', // susceptible, infected, zombie
    }));

    const links = [];
    for (let i = 1; i < N; i++) {
      // Hub attachment logic (mocking scale-free graph)
      const target = Math.random() < 0.2 ? 0 : Math.floor(Math.random() * i);
      links.push({
        source: i,
        target: target,
      });
      // Add random cross-links
      if (Math.random() < 0.05) {
        links.push({
          source: i,
          target: Math.floor(Math.random() * N),
        });
      }
    }
    setGraphData({ nodes, links } as any);
  }, []);

  // Animate the graph based on the simulation infection rate
  useEffect(() => {
    if (graphData.nodes.length === 0) return;
    
    setGraphData((prev: any) => {
      const newNodes = prev.nodes.map((node: any) => {
        // Randomly "infect" nodes proportional to infection rate
        let newStatus = node.status;
        if (node.status === 'susceptible' && Math.random() < infectionRate) {
          newStatus = Math.random() < 0.8 ? 'infected' : 'zombie';
        }
        return { ...node, status: newStatus };
      });
      return { ...prev, nodes: newNodes };
    });
  }, [infectionRate]);

  const nodeColor = useCallback((node: any) => {
    if (node.status === 'infected') return '#ff3333';
    if (node.status === 'zombie') return '#bd10e0';
    // Susceptible nodes colored by community
    const colors = ['#2a4b7c', '#296d98', '#1a365d'];
    return colors[node.group % colors.length];
  }, []);

  return (
    <div className="w-full h-full relative bg-black">
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        nodeColor={nodeColor}
        nodeRelSize={4}
        linkWidth={0.5}
        linkColor={() => 'rgba(255,255,255,0.1)'}
        backgroundColor="#050505"
        enableNodeDrag={false}
        onEngineStop={() => fgRef.current?.zoomToFit(400)}
      />
      <div className="absolute top-4 left-4 text-white text-xs opacity-50">
        LIVE GRAPH TOPOLOGY | {graphData.nodes.length} NODES
      </div>
    </div>
  );
}
