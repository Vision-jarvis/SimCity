"use client";

import { useEffect, useRef, useCallback } from 'react';
import { useStore } from '@/store/useStore';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/stream';

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | undefined>(undefined);
  const { addStreamEvent, setWsConnected, wsConnected } = useStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        console.log('[WS] Connected to event stream');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          addStreamEvent(data);
        } catch (e) {
          console.error('[WS] Parse error:', e);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        console.log('[WS] Disconnected. Reconnecting in 5s...');
        reconnectTimeout.current = setTimeout(connect, 5000);
      };

      ws.onerror = (err) => {
        console.error('[WS] Error:', err);
        ws.close();
      };
    } catch (e) {
      console.error('[WS] Connection failed:', e);
      reconnectTimeout.current = setTimeout(connect, 5000);
    }
  }, [addStreamEvent, setWsConnected]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsConnected(false);
  }, [setWsConnected]);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { wsConnected, connect, disconnect };
}
