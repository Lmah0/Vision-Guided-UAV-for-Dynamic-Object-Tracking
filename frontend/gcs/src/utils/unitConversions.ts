// Distance/Altitude conversions
export const convertDistance = {
  /**
   * Convert meters to feet
   * @param meters - distance in meters
   * @returns distance in feet
   */
  metersToFeet: (meters: number): number => meters * 3.28084,
  
  /**
   * Convert meters to miles
   * @param meters - distance in meters
   * @returns distance in miles
   */
  metersToMiles: (meters: number): number => meters * 0.000621371,
  
  /**
   * Convert meters per second to feet per second
   * @param mps - speed in meters per second
   * @returns speed in feet per second
   */
  mpsToFps: (mps: number): number => mps * 3.28084,
};

// Formatting utilities
export const formatUnits = {
  /**
   * Format speed value with separated value and unit
   * @param mps - speed in meters per second
   * @param isMetric - whether to display in metric units
   * @returns object with formatted value and unit
   */
  speed: (mps: number | undefined | null, isMetric: boolean): { value: string, unit: string } => {
    if (mps === undefined || mps === null || isNaN(mps)) {
      return { value: "0.00", unit: isMetric ? "m/s" : "ft/s" };
    }
    if (isMetric) {
      return { value: mps.toFixed(2), unit: "m/s" };
    } else {
      return { value: convertDistance.mpsToFps(mps).toFixed(2), unit: "ft/s" };
    }
  },
  
  /**
   * Format altitude value with separated value and unit
   * @param meters - distance in meters
   * @param isMetric - whether to display in metric units
   * @returns object with formatted value and unit
   */
  altitude: (meters: number | undefined | null, isMetric: boolean): { value: string, unit: string } => {
    if (meters === undefined || meters === null || isNaN(meters)) {
      return { value: "0.00", unit: isMetric ? "m" : "ft" };
    }
    if (isMetric) {
      return { value: meters.toFixed(2), unit: "m" };
    } else {
      return { value: convertDistance.metersToFeet(meters).toFixed(2), unit: "ft" };
    }
  },
  
  /**
   * Format distance value with appropriate unit (for longer distances)
   * @param meters - distance in meters
   * @param isMetric - whether to display in metric units
   * @returns formatted string with unit
   */
  distance: (meters: number, isMetric: boolean): string => {
    if (meters === undefined || meters === null || isNaN(meters)) {
      return isMetric ? "0.00m" : "0.00 ft";
    }
    if (isMetric) {
      if (meters >= 1000) {
        return `${(meters / 1000).toFixed(2)} km`;
      } else {
        return `${meters.toFixed(2)} m`;
      }
    } else {
      const feet = convertDistance.metersToFeet(meters);
      if (feet >= 5280) {
        return `${convertDistance.metersToMiles(meters).toFixed(2)} mi`;
      } else {
        return `${feet.toFixed(2)} ft`;
      }
    }
  },
  
  /**
   * Format degrees value with degree symbol (for HUD/compact display)
   * @param degrees - value in degrees (angles, coordinates, etc.)
   * @returns formatted string with degree symbol
   */
  degrees: (degrees: number | undefined | null): string => {
    if (degrees === undefined || degrees === null || isNaN(degrees)) {
      return "0.000°";
    }
    return `${degrees.toFixed(3)}°`;
  },
  
  /**
   * Format degrees value with text unit (for detailed FlightData display)
   * @param degrees - value in degrees (angles, coordinates, etc.)
   * @returns object with formatted value and "degrees" unit
   */
  degreesWithTextUnit: (degrees: number | undefined | null): { value: string, unit: string } => {
    if (degrees === undefined || degrees === null || isNaN(degrees)) {
      return { value: "0.0000", unit: "degrees" };
    }
    return {
      value: degrees.toFixed(4),
      unit: "degrees"
    };
  },
  
  /**
   * Universal formatter that handles any telemetry type for FlightData component
   * @param value - raw value to format
   * @param telemetryKey - type of telemetry (speed, altitude, latitude, etc.)
   * @param isMetric - whether to display in metric units
   * @returns object with formatted value and unit
   */
  formatTelemetry: (value: number | undefined | null, telemetryKey: string, isMetric: boolean): { value: string, unit: string } => {
    switch (telemetryKey) {
      case 'speed':
        return formatUnits.speed(value, isMetric);
      case 'altitude':
        return formatUnits.altitude(value, isMetric);
      case 'latitude':
      case 'longitude':
      case 'roll':
      case 'pitch':
      case 'yaw':
        return formatUnits.degreesWithTextUnit(value);
      default:
        if (value === undefined || value === null || isNaN(value)) {
          return { value: "0.00", unit: "" };
        }
        return { value: value.toFixed(2), unit: "" };
    }
  },
};