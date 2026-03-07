"use client";
import { Switch, FormControlLabel, Box, Typography, Paper, Radio, RadioGroup, TextField, InputAdornment, Select, MenuItem, FormControl, Button } from '@mui/material';
import { useState, useEffect } from 'react';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import PublicIcon from '@mui/icons-material/Public';
import StraightenIcon from '@mui/icons-material/Straighten';
import FlightIcon from '@mui/icons-material/Flight';
import StopIcon from '@mui/icons-material/Stop';
import { convertDistance } from '../../../utils/unitConversions';
import { flightModeMapping } from '@/utils/flightModeMapping';
import { useWebSocket } from '@/providers/WebSocketProvider';
import { DroneIcon } from '../../../utils/droneIcon';
import axios from 'axios';

interface ControlsProps {
    showHUDElements: boolean;
    setShowHUDElements: React.Dispatch<React.SetStateAction<boolean>>;
    isMetric: boolean;
    setIsMetric: React.Dispatch<React.SetStateAction<boolean>>;
    followDistance: number;
    setFollowDistance: React.Dispatch<React.SetStateAction<number>>;
}

export default function Controls({ showHUDElements, setShowHUDElements, isMetric, setIsMetric, followDistance, setFollowDistance}: ControlsProps) {
    const { trackingData, aircraftType, flightMode, isRecording, setIsRecording, sendMessage } = useWebSocket();
    const [inputValue, setInputValue] = useState('');
    const isQuadcopter = (aircraftType || '').toLowerCase() === 'quadcopter';

    useEffect(() => {
        const displayDistance = isMetric ? followDistance : convertDistance.metersToFeet(followDistance);
        setInputValue(displayDistance.toString());
    }, [followDistance, isMetric]);

    const handleHudToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
        setShowHUDElements(event.target.checked);
    };
    const handleRecordingToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newRecordingState = event.target.checked;
        setIsRecording(newRecordingState);
        axios.post('http://localhost:8766/recording')
        .catch(error => {
            console.error('Error toggling recording:', error);
            setIsRecording(!newRecordingState);
        });
    };
    const handleMetricToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
        setIsMetric(event.target.value === 'metric');
    };

    const handleFollowDistanceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = event.target.value;
        setInputValue(newValue);
    };

    const commitFollowDistance = async () => {
        try{
            const value = parseFloat(inputValue);
            if (!isNaN(value) && value > 0) {
                const distanceInMeters = isMetric ? value : value / convertDistance.metersToFeet(1);
                setFollowDistance(distanceInMeters);
                const resp = await axios.post(`http://localhost:8766/setFollowDistance/`, {distance: distanceInMeters});
                if (resp.status !== 200) {
                    console.warn('Unexpected response setting follow distance:', resp);
                }
            }
        } catch (error) {
            console.error('Error sending follow distance to backend:', error);
        }
    };

    const handleFlightModeChange = async (event: any) => {
        try {
            const newFlightMode = event.target.value;
            const resp = await axios.post(`http://localhost:8766/setFlightMode`, { mode: newFlightMode });
            if (resp.status !== 200) {
                console.warn('Unexpected response setting flight mode:', resp);
            }
        } catch (error) {
            console.error('Error sending flight mode to backend:', error);
        }
    };

    const handleAircraftTypeChange = (event: any) => {
        const newAircraftType = event.target.value;
        sendMessage(JSON.stringify({
            type: 'set_aircraft_type',
            aircraft_type: newAircraftType,
        }));
    };

    const handleStopFollowing = async () => {
        try{   
            const resp = await axios.post(`http://localhost:8766/stopFollowing/`);
            if (resp.status !== 200) {
                console.warn('Unexpected response stopping following:', resp);
            }
        } catch (error) {
            console.error('Error sending stop following to backend:', error);
        }
    };

    return (
        <div className="w-full h-full p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {/* Stop Following Control */}
                <Paper
                    elevation={3}
                    sx={{
                        p: 2,
                        backgroundColor: 'rgba(38, 38, 38, 0.9)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: 2,
                    }}
                >
                    <Box className="flex items-center mb-2">
                        <StopIcon sx={{ color: trackingData?.tracking ? '#ef4444' : '#6b7280', mr: 1.5, fontSize: 20 }} />
                        <Typography variant="subtitle1" className="text-white font-semibold">
                            Stop Following
                        </Typography>
                    </Box>
                    <Typography variant="caption" className="text-neutral-300 mb-3 block">
                        {trackingData?.tracking ? 'Stop following the current target' : 'No target being followed'}
                    </Typography>
                    <Button
                        id='stop-following-button'
                        variant="contained"
                        onClick={handleStopFollowing}
                        disabled={!trackingData?.tracking}
                        fullWidth
                        sx={{
                            backgroundColor: trackingData?.tracking ? '#ef4444' : '#374151',
                            color: 'white',
                            '&:hover': {
                                backgroundColor: trackingData?.tracking ? '#dc2626' : '#374151',
                            },
                            '&:disabled': {
                                backgroundColor: '#374151',
                                color: '#6b7280',
                            },
                        }}
                    >
                        {trackingData?.tracking ? 'Stop Following' : 'Not Following'}
                    </Button>
                </Paper>

                {/* Follow Distance Control */}
                <Paper
                    elevation={3}
                    sx={{
                        p: 2,
                        backgroundColor: 'rgba(38, 38, 38, 0.9)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: 2,
                    }}
                >
                    <Box className="flex items-center mb-2">
                        <StraightenIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        <Typography variant="subtitle1" className="text-white font-semibold">
                            Follow Distance
                        </Typography>
                    </Box>
                    
                    <Typography variant="caption" className="text-neutral-300 mb-3 block">
                        Minimum distance to maintain from target
                    </Typography>
                    
                    <TextField
                        id='follow-distance-input'
                        type="number"
                        value={inputValue}
                        onChange={handleFollowDistanceChange}
                        onBlur={commitFollowDistance}
                        size="small"
                        InputProps={{
                            endAdornment: (
                                <InputAdornment position="end">
                                    <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                                        {isMetric ? 'meters' : 'feet'}
                                    </Typography>
                                </InputAdornment>
                            ),
                            inputProps: {
                                min: 0.1,
                                step: 1,
                            },
                        }}
                        sx={{
                            '& .MuiOutlinedInput-root': {
                                backgroundColor: 'rgba(55, 65, 81, 0.8)',
                                '& fieldset': {
                                    borderColor: 'rgba(156, 163, 175, 0.3)',
                                },
                                '&:hover fieldset': {
                                    borderColor: 'rgba(156, 163, 175, 0.5)',
                                },
                                '&.Mui-focused fieldset': {
                                    borderColor: '#10b981',
                                },
                                '& input': {
                                    color: 'white',
                                },
                            },
                        }}
                        fullWidth
                    />
                </Paper>

                {/* Flight Mode Control */}
                <Paper
                    elevation={3}
                    sx={{
                        p: 2,
                        backgroundColor: 'rgba(38, 38, 38, 0.9)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: 2,
                    }}
                >
                    <Box className="flex items-center mb-2">
                        {isQuadcopter ? (
                            <DroneIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        ) : (
                            <FlightIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        )}
                        <Typography variant="subtitle1" className="text-white font-semibold">
                            Flight Mode
                        </Typography>
                    </Box>
                    
                    <Typography variant="caption" className="text-neutral-300 mb-3 block">
                        Select aircraft flight mode
                    </Typography>
                    
                    <FormControl fullWidth size="small">
                        <Select
                            id='flight-mode-select'
                            value={flightModeMapping[flightMode as unknown as number]}
                            onChange={handleFlightModeChange}
                            sx={{
                                backgroundColor: 'rgba(55, 65, 81, 0.8)',
                                color: 'white',
                                '& .MuiOutlinedInput-notchedOutline': {
                                    borderColor: 'rgba(156, 163, 175, 0.3)',
                                },
                                '&:hover .MuiOutlinedInput-notchedOutline': {
                                    borderColor: 'rgba(156, 163, 175, 0.5)',
                                },
                                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                                    borderColor: '#10b981',
                                },
                                '& .MuiSelect-icon': {
                                    color: 'white',
                                },
                            }}
                            MenuProps={{
                                PaperProps: {
                                    sx: {
                                        backgroundColor: 'rgba(38, 38, 38, 0.95)',
                                        border: '1px solid rgba(255, 255, 255, 0.2)',
                                        '& .MuiMenuItem-root': {
                                            color: 'white',
                                            '&:hover': {
                                                backgroundColor: 'rgba(55, 65, 81, 0.8)',
                                            },
                                            '&.Mui-selected': {
                                                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                                                '&:hover': {
                                                    backgroundColor: 'rgba(16, 185, 129, 0.3)',
                                                },
                                            },
                                        },
                                    },
                                },
                            }}
                        >
                            {Object.entries(flightModeMapping).filter(([modeId]) => modeId !== '-1').map(([modeId, modeName]) => (
                                <MenuItem key={modeId} value={modeName}>
                                    {modeName}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Paper>

                {/* Aircraft Type Control */}
                <Paper
                    elevation={3}
                    sx={{
                        p: 2,
                        backgroundColor: 'rgba(38, 38, 38, 0.9)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: 2,
                    }}
                >
                    <Box className="flex items-center mb-2">
                        {isQuadcopter ? (
                            <DroneIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        ) : (
                            <FlightIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        )}
                        <Typography variant="subtitle1" className="text-white font-semibold">
                            Aircraft Type
                        </Typography>
                    </Box>

                    <Typography variant="caption" className="text-neutral-300 mb-3 block">
                        Select aircraft type
                    </Typography>

                    <FormControl fullWidth size="small">
                        <Select
                            id='aircraft-type-select'
                            value={aircraftType}
                            onChange={handleAircraftTypeChange}
                            sx={{
                                backgroundColor: 'rgba(55, 65, 81, 0.8)',
                                color: 'white',
                                '& .MuiOutlinedInput-notchedOutline': {
                                    borderColor: 'rgba(156, 163, 175, 0.3)',
                                },
                                '&:hover .MuiOutlinedInput-notchedOutline': {
                                    borderColor: 'rgba(156, 163, 175, 0.5)',
                                },
                                '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                                    borderColor: '#10b981',
                                },
                                '& .MuiSelect-icon': {
                                    color: 'white',
                                },
                            }}
                            MenuProps={{
                                PaperProps: {
                                    sx: {
                                        backgroundColor: 'rgba(38, 38, 38, 0.95)',
                                        border: '1px solid rgba(255, 255, 255, 0.2)',
                                        '& .MuiMenuItem-root': {
                                            color: 'white',
                                            '&:hover': {
                                                backgroundColor: 'rgba(55, 65, 81, 0.8)',
                                            },
                                            '&.Mui-selected': {
                                                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                                                '&:hover': {
                                                    backgroundColor: 'rgba(16, 185, 129, 0.3)',
                                                },
                                            },
                                        },
                                    },
                                },
                            }}
                        >
                            <MenuItem value="plane">Plane</MenuItem>
                            <MenuItem value="quadcopter">Quadcopter</MenuItem>
                        </Select>
                    </FormControl>
                </Paper>

                {/* Recording Control */}
                <Paper
                    elevation={3}
                    sx={{
                        p: 2,
                        backgroundColor: 'rgba(38, 38, 38, 0.9)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: 2,
                    }}
                >
                    <Box className="flex items-center mb-2">
                        <RadioButtonCheckedIcon sx={{ color: isRecording ? '#ef4444' : 'white', mr: 1.5, fontSize: 20 }} />
                        <Typography variant="subtitle1" className="text-white font-semibold">
                            Record Objects
                        </Typography>
                    </Box>
                    
                    <Typography variant="caption" className="text-neutral-300 mb-3 block">
                        Record tracked objects
                    </Typography>
                    
                    <FormControlLabel
                        control={
                            <Switch
                                id='record-switch'
                                checked={isRecording}
                                onChange={handleRecordingToggle}
                                size="small"
                                sx={{
                                    '& .MuiSwitch-switchBase.Mui-checked': {
                                        color: '#ef4444',
                                    },
                                    '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                        backgroundColor: '#ef4444',
                                    },
                                    '& .MuiSwitch-track': {
                                        backgroundColor: '#6b7280',
                                    },
                                }}
                            />
                        }
                        label={
                            <Typography variant="body2" className="text-white">
                                {isRecording ? 'Recording' : 'Stopped'}
                            </Typography>
                        }
                    />
                </Paper>

                {/* HUD Elements Control */}
                <Paper
                    elevation={3}
                    sx={{
                        p: 2,
                        backgroundColor: 'rgba(38, 38, 38, 0.9)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: 2,
                    }}
                >
                    <Box className="flex items-center mb-2">
                        {showHUDElements ? (
                            <VisibilityIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        ) : (
                            <VisibilityOffIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        )}
                        <Typography variant="subtitle1" className="text-white font-semibold">
                            HUD Elements
                        </Typography>
                    </Box>
                    
                    <Typography variant="caption" className="text-neutral-300 mb-3 block">
                        Toggle HUD overlay visibility
                    </Typography>
                    
                    <FormControlLabel
                        control={
                            <Switch
                                id='hud-elements-toggle'
                                checked={showHUDElements}
                                onChange={handleHudToggle}
                                size="small"
                                sx={{
                                    '& .MuiSwitch-switchBase.Mui-checked': {
                                        color: '#3b82f6',
                                    },
                                    '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                        backgroundColor: '#3b82f6',
                                    },
                                    '& .MuiSwitch-track': {
                                        backgroundColor: '#6b7280',
                                    },
                                }}
                            />
                        }
                        label={
                            <Typography variant="body2" className="text-white">
                                {showHUDElements ? 'Visible' : 'Hidden'}
                            </Typography>
                        }
                    />
                </Paper>

                {/* Units Control */}
                <Paper
                    elevation={3}
                    sx={{
                        p: 2,
                        backgroundColor: 'rgba(38, 38, 38, 0.9)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: 2,
                    }}
                >
                    <Box className="flex items-center mb-2">
                        <PublicIcon sx={{ color: 'white', mr: 1.5, fontSize: 20 }} />
                        <Typography variant="subtitle1" className="text-white font-semibold">
                            Units
                        </Typography>
                    </Box>
                    
                    <Typography variant="caption" className="text-neutral-300 mb-3 block">
                        Display units system
                    </Typography>
                    
                    <RadioGroup
                        id='unit-toggle'
                        value={isMetric ? 'metric' : 'imperial'}
                        onChange={handleMetricToggle}
                        row
                    >
                        <FormControlLabel
                            value="metric"
                            control={
                                <Radio
                                    id='metric-radio'
                                    size="small"
                                    sx={{
                                        color: '#6b7280',
                                        '&.Mui-checked': {
                                            color: '#10b981',
                                        },
                                    }}
                                />
                            }
                            label={
                                <Typography variant="body2" className="text-white">
                                    Metric
                                </Typography>
                            }
                        />
                        <FormControlLabel
                            value="imperial"
                            control={
                                <Radio
                                    id='imperial-radio'
                                    size="small"
                                    sx={{
                                        color: '#6b7280',
                                        '&.Mui-checked': {
                                            color: '#10b981',
                                        },
                                    }}
                                />
                            }
                            label={
                                <Typography variant="body2" className="text-white">
                                    Imperial
                                </Typography>
                            }
                        />
                    </RadioGroup>
                </Paper>
            </div>
        </div>
    );
}