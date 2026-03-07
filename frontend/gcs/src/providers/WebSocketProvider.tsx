'use client';
import { createContext, useContext, useState, useEffect, ReactNode, useRef } from 'react';
import { TelemetryData, trackingData } from '@/utils/telemetryConfig';
import axios from 'axios';

interface WebSocketContextType {
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  sendMessage: (message: string) => void;
  telemetryData: any;
  batteryData: any;
  droneConnection: boolean;
  trackingData: any;
  flightMode: number;
  aircraftType: string;
  isRecording: boolean;
  setIsRecording: (value: boolean) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}

interface WebSocketProviderProps {
  children: ReactNode;
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [telemetryData, setTelemetryData] = useState<TelemetryData | null>(null);
  const [batteryData, setBatteryData] = useState<any>(null);
  const [droneConnection, setDroneConnection] = useState<boolean>(false);
  const [trackingData, setTrackingData] = useState<trackingData | null>(null);
  const [flightMode, setFlightMode] = useState<number>(-1);
  const [aircraftType, setAircraftType] = useState<string>('quadcopter');
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const droneTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = () => {
    // Don't try to connect if already connecting or connected
    if (connectionStatus === 'connecting' || connectionStatus === 'connected') {
      return;
    }

    // Stop trying after max attempts
    if (reconnectAttempts.current >= maxReconnectAttempts) {
      console.log('Max reconnection attempts reached');
      setConnectionStatus('error');
      return;
    }

    setConnectionStatus('connecting');
    
    const websocket = new WebSocket('ws://localhost:8766/ws/gcs');
    wsRef.current = websocket;

    websocket.onopen = () => {
      setConnectionStatus('connected');
      reconnectAttempts.current = 0;
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);    
        if (data) 
        {
          if(data.status)
          {
            if(data.status !== 200) {
              console.log('Error from server:', data.error || 'Unknown error');
            }
            return;
          }
          // Update telemetry data
          setTelemetryData({ 
              speed: Math.sqrt((data.dlat ** 2) + (data.dlon ** 2) + (data.dalt ** 2)),
              altitude: data.altitude,
              latitude: data.latitude,
              longitude: data.longitude,
              roll: data.roll,
              pitch: data.pitch,
              yaw: data.yaw });

          setFlightMode(data.flight_mode);
          if (data.aircraft_type) {
            setAircraftType(data.aircraft_type);
          }
          
          if (data.battery_voltage && data.battery_remaining) {
            setBatteryData({
              percentage: data.battery_remaining,
              usage: data.battery_voltage
            });
          }
          
          // Connection is considered active if we're receiving telemetry
          setDroneConnection(true);
          setTrackingData({ tracking: data.tracking, tracked_class: data.tracked_class, distance_to_target: data.distance_to_target })
          setIsRecording(data.is_recording);

          // Reset drone connection timeout - if we don't receive data for 5 seconds, mark as disconnected
          if (droneTimeoutRef.current) {
            clearTimeout(droneTimeoutRef.current);
          }
          droneTimeoutRef.current = setTimeout(() => {
            setDroneConnection(false);
          }, 5000);
        }
      } catch (error) {
        console.log('Received text:', event.data);
      }
    };

    websocket.onclose = () => {
      setConnectionStatus('disconnected');  
      // Try to reconnect after 1 second
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        console.log(`Attempting Reconnect... (${reconnectAttempts.current}/${maxReconnectAttempts})`);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, 1000);
      }
    };

    websocket.onerror = (error) => {
      setConnectionStatus('error');
    };
  };

  const sendMessage = (message: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(message);
    }
  };

  useEffect(() => {
    connect(); 
    return () => {
      // Clear timeouts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (droneTimeoutRef.current) {
        clearTimeout(droneTimeoutRef.current);
      }
      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const value: WebSocketContextType = { 
    connectionStatus, 
    sendMessage,
    telemetryData, 
    batteryData, 
    droneConnection,
    trackingData,
    flightMode,
    aircraftType,
    isRecording,
    setIsRecording
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}