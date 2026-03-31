import { MapConfig, AnimationConfig, TerrainConfig, SkyConfig } from "@/utils/types";

export const DEFAULT_MAP_CONFIG: MapConfig = {
  center: [-74.5, 40],
  zoom: 16,
  pitch: 60,
  bearing: 0,
  style: "mapbox://styles/mapbox/streets-v11",
};

export const DEFAULT_ANIMATION_CONFIG: AnimationConfig = {
  stepsPerSegment: 60,
  hotspotsInterval: 40,
  cameraOffset: 0.0002,
  cameraDuration: 200,
};

export const DEFAULT_TERRAIN_CONFIG: TerrainConfig = {
  source: "mapbox-dem",
  exaggeration: 1.5,
};

export const DEFAULT_SKY_CONFIG: SkyConfig = {
  sunPosition: [0.0, 0.0],
  sunIntensity: 15,
};