import ConnectionStatus from './components/ConnectionStatus';
import ServerConnection from './components/ServerConnection';
import BatteryGuage from './components/Battery';
import VideoFeed from './components/VideoFeed';
import TelemetryData from './components/TelemetryData';
import FlightMode from './components/FlightMode';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import { useWebSocket } from '@/providers/WebSocketProvider';

interface HUDProps {
    showHUDElements: boolean;
    pinnedTelemetry: string[];
    isMetric: boolean;
}

export default function HUD({ showHUDElements, pinnedTelemetry, isMetric }: HUDProps) {
    const {droneConnection, isRecording} = useWebSocket();
    return (
        <div id='HUD' className="w-full h-full relative">
            <div id='video-feed' className="absolute inset-0 z-0">
                <VideoFeed />
            </div>
            <div id='server-connection' className="absolute top-4 left-4 z-50">
                <ServerConnection />
            </div>
            <div id='drone-connection' className="absolute top-10 left-4 z-10">
                <ConnectionStatus />
            </div>
            {isRecording && (
                <div id='recording' className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 flex items-center space-x-1 px-2 py-1">
                    <RadioButtonCheckedIcon className="text-red-600 animate-pulse" />
                </div>
            )}
            {/* HUD Area Overlay */}
            {showHUDElements && (
                <div id='HUD-elements'>
                    <div className="absolute inset-2 pointer-events-none z-5">
                        <div id='overlay' className={`w-full h-full rounded-xl ${
                            droneConnection 
                                ? 'bg-black/10 border border-white/20' 
                                : 'bg-red-500/10 border border-red-500/80 animate-pulse'
                        }`}></div>
                    </div>
                    
                    <div id='HUD-telemetry' className="absolute bottom-4 right-4 z-10">
                        <TelemetryData pinnedTelemetry={pinnedTelemetry} isMetric={isMetric} />
                    </div>
                    <div id='HUD-flight-mode' className="absolute bottom-4 left-4 z-10">
                        <FlightMode isMetric={isMetric} />
                    </div>
                    <div id='HUD-battery' className="absolute top-4 right-4 z-10">
                        <BatteryGuage />
                    </div>
                </div>
            )}
        </div>
    );
}