"use client";
import { useRef, useCallback } from "react";
import mapboxgl from "mapbox-gl";
import { TelemetryPoint, Coordinates, RoutePoint, AnimationConfig } from "@/utils/types";
import { calculateBearing, offsetBehind, interpolateCoordinates, calculateDistance } from "@/utils/mapUtils";

interface UseMapAnimationProps {
  map: React.RefObject<mapboxgl.Map | null>;
  marker: React.RefObject<mapboxgl.Marker | null>;
  pathCoordinates: Coordinates[];
  telemetryData: TelemetryPoint[];
  config: AnimationConfig;
  onAnimationEnd?: () => void;
}

export const useMapAnimation = ({map, marker, pathCoordinates, telemetryData, config, onAnimationEnd}: UseMapAnimationProps) => {
  const animReq = useRef<number | null>(null);
  const animationState = useRef({
    i: 0,
    progress: 0,
    points: [] as RoutePoint[],
    isRunning: false,
    isPaused: false,
  });

  const runAnimation = useCallback(() => {
    if (!map.current || !marker.current) return;

    const routeSource = map.current.getSource("route") as mapboxgl.GeoJSONSource;
    const pointsSource = map.current.getSource("route-points") as mapboxgl.GeoJSONSource;

    // Reset animation state only if starting fresh
    if (!animationState.current.isPaused) {
      animationState.current = {
        i: 0,
        progress: 0,
        points: [],
        isRunning: true,
        isPaused: false,
      };

      // Reset visual state
      routeSource.setData({
        type: "Feature",
        geometry: { type: "LineString", coordinates: [pathCoordinates[0]] },
        properties: {},
      });
      pointsSource.setData({ type: "FeatureCollection", features: [] });
      marker.current.setLngLat(pathCoordinates[0]);
    } else {
      // Resume from paused state
      animationState.current.isRunning = true;
      animationState.current.isPaused = false;
    }

    const animate = () => {
      if (!animationState.current.isRunning) return;
      
      const { i, progress, points } = animationState.current;
      
      if (i >= pathCoordinates.length - 1) {
        animationState.current.isRunning = false;
        onAnimationEnd?.();
        return;
      }

      const start = pathCoordinates[i];
      const end = pathCoordinates[i + 1];
      const progressRatio = progress / config.stepsPerSegment;
      const currentPos = interpolateCoordinates(start, end, progressRatio);
      
      marker.current!.setLngLat(currentPos);

      // Update route line
      const currentData = (routeSource as any)._data;
      const coords = currentData.geometry.coordinates.concat([currentPos]);
      routeSource.setData({
        type: "Feature",
        geometry: { type: "LineString", coordinates: coords },
        properties: {},
      });

      // Add hotspot points at intervals
      if (progress % config.hotspotsInterval === 0) {
        const bearing = calculateBearing(start, end);
        const speed = calculateDistance(start, end);
        
        // Get telemetry data for this point if available
        const telemetryIndex = Math.floor((i + progressRatio) * (telemetryData?.length || 0) / pathCoordinates.length);
        const telemetryPoint = telemetryData?.[telemetryIndex];
        
        const newPoint: RoutePoint = {
          type: "Feature",
          id: points.length, // Add unique ID for feature-state functionality
          geometry: { type: "Point", coordinates: currentPos },
          properties: { 
            position: currentPos, 
            speed: telemetryPoint?.speed?.toFixed(1) || speed.toFixed(4),
            timestamp: telemetryPoint ? new Date(telemetryPoint.timestamp).getTime() : Date.now(),
            pointIndex: points.length + 1,
            telemetryIndex: telemetryIndex, // Just store the index to reference the original data
          },
        };
        animationState.current.points.push(newPoint);
        pointsSource.setData({ type: "FeatureCollection", features: animationState.current.points });
      }

      // Update camera position
      const bearing = calculateBearing(start, end);
      const cameraPos = offsetBehind(currentPos, bearing, config.cameraOffset);
      map.current!.easeTo({
        center: cameraPos,
        zoom: 20,
        pitch: 45,
        bearing,
        duration: config.cameraDuration,
        easing: (t) => t,
      });

      animationState.current.progress++;
      if (animationState.current.progress > config.stepsPerSegment) {
        animationState.current.progress = 0;
        animationState.current.i++;
      }

      animReq.current = requestAnimationFrame(animate);
    };

    animate();
  }, [map, marker, pathCoordinates, config]);

  const stopAnimation = useCallback(() => {
    if (animReq.current) {
      cancelAnimationFrame(animReq.current);
      animReq.current = null;
    }
    animationState.current.isRunning = false;
    animationState.current.isPaused = false;
  }, []);

  const pauseAnimation = useCallback(() => {
    if (animReq.current) {
      cancelAnimationFrame(animReq.current);
      animReq.current = null;
    }
    animationState.current.isRunning = false;
    animationState.current.isPaused = true;
  }, []);

  const resumeAnimation = useCallback(() => {
    if (animationState.current.isPaused) {
      runAnimation();
    }
  }, [runAnimation]);

  const restartAnimation = useCallback(() => {
    stopAnimation();
    animationState.current.isPaused = false;
    runAnimation();
  }, [stopAnimation, runAnimation]);

  const skipAnimation = useCallback(() => {
    if (!map.current || !marker.current) return;

    const routeSource = map.current.getSource("route") as mapboxgl.GeoJSONSource;
    const pointsSource = map.current.getSource("route-points") as mapboxgl.GeoJSONSource;

    // Stop any running animation
    stopAnimation();

    // Set animation state to completed
    animationState.current = {
      i: pathCoordinates.length - 1,
      progress: 0,
      points: [],
      isRunning: false,
      isPaused: false,
    };

    // Move marker to final position
    const finalPosition = pathCoordinates[pathCoordinates.length - 1];
    marker.current.setLngLat(finalPosition);

    // Display the complete route
    routeSource.setData({
      type: "Feature",
      geometry: { type: "LineString", coordinates: pathCoordinates },
      properties: {},
    });

    // Generate all hotspot points for the complete route
    const allPoints: RoutePoint[] = [];
    for (let i = 0; i < pathCoordinates.length - 1; i++) {
      for (let progress = 0; progress <= config.stepsPerSegment; progress += config.hotspotsInterval) {
        const start = pathCoordinates[i];
        const end = pathCoordinates[i + 1];
        const progressRatio = progress / config.stepsPerSegment;
        const currentPos = interpolateCoordinates(start, end, progressRatio);
        const speed = calculateDistance(start, end);
        
        // Get telemetry data for this point if available
        const telemetryIndex = Math.floor((i + progressRatio) * (telemetryData?.length || 0) / pathCoordinates.length);
        const telemetryPoint = telemetryData?.[telemetryIndex];
        
        const point: RoutePoint = {
          type: "Feature",
          id: allPoints.length,
          geometry: { type: "Point", coordinates: currentPos },
          properties: { 
            position: currentPos, 
            speed: telemetryPoint?.speed?.toFixed(1) || speed.toFixed(4),
            timestamp: telemetryPoint ? new Date(telemetryPoint.timestamp).getTime() : Date.now(),
            pointIndex: allPoints.length + 1,
            telemetryIndex: telemetryIndex,
          },
        };
        allPoints.push(point);
      }
    }

    // Update points source with all hotspots
    pointsSource.setData({ type: "FeatureCollection", features: allPoints });

    // Set camera to show the complete route
    const bounds = new mapboxgl.LngLatBounds();
    pathCoordinates.forEach(coord => bounds.extend(coord));
    
    map.current.fitBounds(bounds, {
      padding: 100,
      maxZoom: 14,
      duration: 1000,
    });

    // Call animation end callback
    onAnimationEnd?.();
  }, [map, marker, pathCoordinates, config, telemetryData, stopAnimation, onAnimationEnd]);

  const getAnimationStatus = useCallback(() => ({
    isRunning: animationState.current.isRunning,
    isPaused: animationState.current.isPaused,
    progress: animationState.current.i / (pathCoordinates.length - 1),
  }), [pathCoordinates.length]);

  return {runAnimation, stopAnimation, pauseAnimation, resumeAnimation, restartAnimation, skipAnimation, getAnimationStatus};
};