import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import InfoDashBoard from '../src/components/InfoDashBoard/InfoDashBoard';
import { useWebSocket } from '../src/providers/WebSocketProvider';
import axios from 'axios';

jest.mock('axios');
jest.mock('../src/providers/WebSocketProvider', () => ({
  useWebSocket: jest.fn(),
  sendMessage: jest.fn()
}));

// Mock console errors for now to reduce noise during tests
console.error = jest.fn();

// Reset mocks before each test
beforeEach(() => {
  jest.clearAllMocks();
  const mockSendMessage = jest.fn();
  useWebSocket.mockReturnValue({
    connectionStatus: 'connected',
    droneConnection: true,
    telemetryData: null,
    isRecording: false,
    batteryData: null,
    trackingData: { tracking: false, tracked_class: null },
    flightMode: 3,
    aircraftType: 'quadcopter',
    sendMessage: mockSendMessage,
  });
  axios.post.mockResolvedValue({ status: 200, data: {} });
  axios.get.mockResolvedValue({ status: 200, data: [] });
});

// Clean up after each test
afterEach(() => {
  cleanup();
});

const mockProps = {
  showHUDElements: true,
  setShowHUDElements: jest.fn(),
  isMetric: true,
  setIsMetric: jest.fn(),
  pinnedTelemetry: [],
  setPinnedTelemetry: jest.fn(),
  followDistance: 10,
  setFollowDistance: jest.fn(),
  flightMode: 3,
  setFlightMode: jest.fn()
};

const mock_telemetry = {
  speed: 15,
  altitude: 1200,
  latitude: 37.7749,
  longitude: -122.4194,
  roll: -1.5,
  pitch: 0.6,
  yaw: 2.5
};

const test_main_container = () => {
  const container = document.getElementById('info-dashboard');
  expect(container).toBeInTheDocument();
}

test('Tabs Render and are Selectable', () => {
  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  const flightTelemetryTab = screen.getByRole('tab', { name: /flight telemetry/i });
  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  const recordedObjectsTab = screen.getByRole('tab', { name: /recorded objects/i });

  expect(flightTelemetryTab).toHaveClass('Mui-selected'); // selected tab
  expect(controlsTab).not.toHaveClass('Mui-selected');
  expect(recordedObjectsTab).not.toHaveClass('Mui-selected');

  // Simulate clicking on Controls tab
  fireEvent.click(controlsTab);
  expect(controlsTab).toHaveClass('Mui-selected');
  expect(flightTelemetryTab).not.toHaveClass('Mui-selected');
  expect(recordedObjectsTab).not.toHaveClass('Mui-selected');

  // Simulate clicking on Recorded Objects tab
  fireEvent.click(recordedObjectsTab);
  expect(recordedObjectsTab).toHaveClass('Mui-selected');
  expect(flightTelemetryTab).not.toHaveClass('Mui-selected');
  expect(controlsTab).not.toHaveClass('Mui-selected');
});

test('Telemetry Tab Displays Data', () => {
  useWebSocket.mockReturnValue({ 
    telemetryData: mock_telemetry,
    trackingData: { tracking: false, tracked_class: null },
    flightMode: 3
  });
  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  const flightTelemetryTab = screen.getByRole('tab', { name: /flight telemetry/i });
  fireEvent.click(flightTelemetryTab);

  const speed = document.getElementById('info-dashboard-telemetry-speed');
  const altitude = document.getElementById('info-dashboard-telemetry-altitude');
  const latitude = document.getElementById('info-dashboard-telemetry-latitude');
  const longitude = document.getElementById('info-dashboard-telemetry-longitude');
  const roll = document.getElementById('info-dashboard-telemetry-roll');
  const pitch = document.getElementById('info-dashboard-telemetry-pitch');
  const yaw = document.getElementById('info-dashboard-telemetry-yaw');

  expect(speed).toBeInTheDocument();
  expect(altitude).toBeInTheDocument();
  expect(latitude).toBeInTheDocument();
  expect(longitude).toBeInTheDocument();
  expect(roll).toBeInTheDocument();
  expect(pitch).toBeInTheDocument();
  expect(yaw).toBeInTheDocument();

  expect(speed).toHaveTextContent(mock_telemetry.speed.toFixed(2).toString());
  expect(altitude).toHaveTextContent(mock_telemetry.altitude.toFixed(2).toString());
  expect(latitude).toHaveTextContent(mock_telemetry.latitude.toFixed(4).toString());
  expect(longitude).toHaveTextContent(mock_telemetry.longitude.toFixed(4).toString());
  expect(roll).toHaveTextContent(mock_telemetry.roll.toFixed(4).toString());
  expect(pitch).toHaveTextContent(mock_telemetry.pitch.toFixed(4).toString());
  expect(yaw).toHaveTextContent(mock_telemetry.yaw.toFixed(4).toString());
});

test('Test Unpinning Telemetry', () => {
  useWebSocket.mockReturnValue({
    telemetryData: mock_telemetry,
    trackingData: { tracking: false, tracked_class: null },
    flightMode: 3
  });
  const propsWithPinned = {...mockProps, pinnedTelemetry: ['speed', 'altitude']};
  render(<InfoDashBoard {...propsWithPinned} />);
  test_main_container();

  const speedButton = document.getElementById('pin-button-speed');
  const altitudeButton = document.getElementById('pin-button-altitude');
  expect(speedButton).toBeInTheDocument();
  expect(altitudeButton).toBeInTheDocument();

  // Pinned items should have gold color
  expect(speedButton).toHaveStyle('color: rgb(251, 191, 36)');
  expect(altitudeButton).toHaveStyle('color: rgb(251, 191, 36)');

  // Test click behavior (unpinning speed)
  fireEvent.click(speedButton);
  expect(propsWithPinned.setPinnedTelemetry).toHaveBeenCalledWith(['altitude']);
});

test('Test Pinning Telemetry', () => {
  useWebSocket.mockReturnValue({
    telemetryData: mock_telemetry,
    trackingData: { tracking: false, tracked_class: null },
    flightMode: 3
  });
  const propsUnpinned = {...mockProps, pinnedTelemetry: []};
  render(<InfoDashBoard {...propsUnpinned} />);

  const speedButton = document.getElementById('pin-button-speed');
  const altitudeButton = document.getElementById('pin-button-altitude');
  
  // Unpinned items should have gray color
  expect(speedButton).toHaveStyle('color: rgba(255, 255, 255, 0.6)');
  expect(altitudeButton).toHaveStyle('color: rgba(255, 255, 255, 0.6)');

  // Test adding pins (pinning speed)
  fireEvent.click(speedButton);
  expect(propsUnpinned.setPinnedTelemetry).toHaveBeenCalledWith(['speed']);
});

test('Controls Tab Displays Controls', () => {
  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  // Check for existence of key controls elements
  const stopFollowingButton = document.getElementById('stop-following-button');
  expect(stopFollowingButton).toBeInTheDocument();

  const recordButton = document.getElementById('record-switch');
  expect(recordButton).toBeInTheDocument();

  const metricRadio = document.getElementById('metric-radio');
  const imperialRadio = document.getElementById('imperial-radio');
  expect(metricRadio).toBeInTheDocument();
  expect(imperialRadio).toBeInTheDocument();

  const followDistanceInput = document.getElementById('follow-distance-input');
  expect(followDistanceInput).toBeInTheDocument();

  const flightModeSelect = document.getElementById('flight-mode-select');
  expect(flightModeSelect).toBeInTheDocument();

  const aircraftTypeSelect = document.getElementById('aircraft-type-select');
  expect(aircraftTypeSelect).toBeInTheDocument();

  const hudToggle = document.getElementById('hud-elements-toggle');
  expect(hudToggle).toBeInTheDocument();
});

test('Toggle Recording', async () => {
  const mockSendMessage = jest.fn();
  useWebSocket.mockReturnValue({
    telemetryData: null,
    isRecording: false,
    trackingData: { tracking: false, tracked_class: null },
    flightMode: 3,
    aircraftType: 'quadcopter',
    sendMessage: mockSendMessage,
    setIsRecording: jest.fn()
  });

  // Mock axios response for recording toggle
  axios.post.mockResolvedValue({
    status: 200,
    data: { is_recording: true }
  });

  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  const recordButton = document.getElementById('record-switch');
  expect(recordButton).toBeInTheDocument();

  // Simulate clicking the record button
  fireEvent.click(recordButton);

  // Wait for the async axios call to complete
  await new Promise(resolve => setTimeout(resolve, 100));

  expect(axios.post).toHaveBeenCalledWith('http://localhost:8766/recording');
});

test('Imperial to Metric Toggle', () => {
  const imperialProps = { ...mockProps, isMetric: false };
  render(<InfoDashBoard {...imperialProps} />);
  test_main_container();

  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  // Get the metric and imperial radios directly
  const metricRadio = document.getElementById('metric-radio');
  const imperialRadio =document.getElementById('imperial-radio');

  expect(metricRadio).toBeInTheDocument();
  expect(imperialRadio).toBeInTheDocument();

  expect(imperialRadio).toBeChecked();
  expect(metricRadio).not.toBeChecked();

  // Click Metric. should setIsMetric(true)
  fireEvent.click(metricRadio);
  expect(imperialProps.setIsMetric).toHaveBeenLastCalledWith(true);
});

test('Follow Distance Adjustment', () => {
  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  const followDistanceInput = document.getElementById('follow-distance-input');
  expect(followDistanceInput).toBeInTheDocument();
  expect(followDistanceInput.value).toBe(mockProps.followDistance.toString());

  // Change value to 20
  fireEvent.change(followDistanceInput, { target: { value: '20' } });
  expect(followDistanceInput.value).toBe('20');

  // Simulate blur event to commit change
  fireEvent.blur(followDistanceInput);
  expect(mockProps.setFollowDistance).toHaveBeenLastCalledWith(20);
});

test('Flight Mode Selection', async () => {
  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  // Go to Controls tab
  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  // Find the flight mode select component
  const flightModeSelect = document.getElementById('flight-mode-select');
  expect(flightModeSelect).toBeInTheDocument();

  // Check that the current flight mode is displayed (mocked as 3 = 'Auto')
  expect(screen.getByText('Auto')).toBeInTheDocument();

  // Open dropdown
  fireEvent.mouseDown(flightModeSelect);

  // Select "Loiter" (different from current mode)
  const loiterOption = screen.getByRole('option', { name: 'Loiter' });
  fireEvent.click(loiterOption);
  
  // Wait for async axios call to complete
  await screen.findByText('Auto'); // Waits for re-render
  
  // Assert that axios.post was called with the correct flight mode value
  expect(axios.post).toHaveBeenCalledWith(
    'http://localhost:8766/setFlightMode',
    { mode: 'Loiter' }
  );
});

test('Toggle HUD Visibility Off', () => {
  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  const hudToggle = document.getElementById('hud-elements-toggle');
  expect(hudToggle).toBeInTheDocument();

  // Initial state is true
  expect(hudToggle).toBeChecked();

  // Click to toggle off
  fireEvent.click(hudToggle);
  expect(mockProps.setShowHUDElements).toHaveBeenLastCalledWith(false);
});

test('Toggle HUD Visibility On', () => {
  const hudOffProps = { ...mockProps, showHUDElements: false };
  render(<InfoDashBoard {...hudOffProps} />);
  test_main_container();

  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  const hudToggle = document.getElementById('hud-elements-toggle');
  expect(hudToggle).toBeInTheDocument();

  // Initial state is true
  expect(hudToggle).not.toBeChecked();

  // Click to toggle off
  fireEvent.click(hudToggle);
  expect(mockProps.setShowHUDElements).toHaveBeenLastCalledWith(true);
});

test('Aircraft Type Selection Sends WebSocket Command', () => {
  const mockSendMessage = jest.fn();
  useWebSocket.mockReturnValue({
    connectionStatus: 'connected',
    droneConnection: true,
    telemetryData: null,
    isRecording: false,
    batteryData: null,
    trackingData: { tracking: false, tracked_class: null },
    flightMode: 3,
    aircraftType: 'quadcopter',
    sendMessage: mockSendMessage,
  });

  render(<InfoDashBoard {...mockProps} />);
  test_main_container();

  const controlsTab = screen.getByRole('tab', { name: /controls/i });
  fireEvent.click(controlsTab);

  const aircraftTypeSelect = document.getElementById('aircraft-type-select');
  expect(aircraftTypeSelect).toBeInTheDocument();

  fireEvent.mouseDown(aircraftTypeSelect);
  const planeOption = screen.getByRole('option', { name: 'Plane' });
  fireEvent.click(planeOption);

  expect(mockSendMessage).toHaveBeenCalledWith(
    JSON.stringify({
      type: 'set_aircraft_type',
      aircraft_type: 'plane',
    })
  );
});