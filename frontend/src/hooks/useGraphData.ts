"use client";

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface GraphData {
  nodes: any[];
  edges: any[];
  stats: { total_nodes: number; total_edges: number };
}

export function useGraphData() {
  const [data, setData] = useState<GraphData>({
    nodes: [],
    edges: [],
    stats: { total_nodes: 0, total_edges: 0 },
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchNodes = useCallback(async (label?: string, limit = 100) => {
    try {
      const params = new URLSearchParams();
      if (label) params.set('label', label);
      params.set('limit', String(limit));
      const res = await axios.get(`${API_URL}/graph/nodes?${params}`);
      return res.data;
    } catch (e: any) {
      console.error('Failed to fetch nodes:', e.message);
      return [];
    }
  }, []);

  const fetchEdges = useCallback(async (type?: string, limit = 100) => {
    try {
      const params = new URLSearchParams();
      if (type) params.set('type', type);
      params.set('limit', String(limit));
      const res = await axios.get(`${API_URL}/graph/edges?${params}`);
      return res.data;
    } catch (e: any) {
      console.error('Failed to fetch edges:', e.message);
      return [];
    }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/graph/stats`);
      return res.data;
    } catch (e: any) {
      console.error('Failed to fetch stats:', e.message);
      return { total_nodes: 0, total_edges: 0 };
    }
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nodes, edges, stats] = await Promise.all([
        fetchNodes(),
        fetchEdges(),
        fetchStats(),
      ]);
      setData({ nodes, edges, stats });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [fetchNodes, fetchEdges, fetchStats]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { data, loading, error, refresh, fetchNodes, fetchEdges, fetchStats };
}
