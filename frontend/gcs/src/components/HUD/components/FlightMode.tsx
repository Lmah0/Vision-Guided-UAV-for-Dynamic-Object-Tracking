"use client";
import { Paper, Typography, Box } from "@mui/material";
import FlightIcon from '@mui/icons-material/Flight';
import { formatUnits } from "../../../utils/unitConversions";
import { useWebSocket } from "../../../providers/WebSocketProvider";
import { flightModeMapping } from "@/utils/flightModeMapping";
import { DroneIcon } from "@/utils/droneIcon";

interface FlightModeProps {
    isMetric: boolean;
}

export default function FlightMode({ isMetric }: FlightModeProps) {
    const { trackingData, aircraftType, flightMode } = useWebSocket();
    const isQuadcopter = (aircraftType || '').toLowerCase() === 'quadcopter';

    return (
        <Paper
            elevation={3}
            sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                p: 2,
                borderRadius: 4,
                backgroundColor: "rgba(0,0,0,0.6)",
                border: "1px solid rgba(255,255,255,0.3)",
                color: "white",
                minWidth: 300,
                minHeight: 70,
            }}
        >
            <Box className="flex items-center gap-1.5">
                {isQuadcopter ? (
                    <DroneIcon sx={{ color: '#3b82f6', fontSize: 30 }} />
                ) : (
                    <FlightIcon sx={{ color: '#3b82f6', fontSize: 30 }} />
                )}
                <Typography id='flight-mode' className="text-white font-bold">
                    {flightModeMapping[flightMode as unknown as number]}
                </Typography>
            </Box>

            {trackingData?.tracking ? (
                <Box className="flex flex-col items-end">
                    <Box className="flex items-center gap-1.5">
                        <Typography variant="caption" className="text-neutral-400 text-xs">
                            Tracking:
                        </Typography>
                        <Typography variant="caption" className="font-semibold text-sm">
                            {trackingData.tracked_class?.toUpperCase() || 'UNKNOWN'}
                        </Typography>
                    </Box>
                </Box>
            ) : (
                <Typography variant="body2" className="text-neutral-500">
                    No tracking
                </Typography>
            )}
        </Paper>
    );
}