import React, { useState, useEffect, useRef, useCallback } from 'react';

export default function VideoFeed() {
    const backendPort = process.env.REACT_APP_BACKEND_PORT || 8766;
    const webrtcUrl = `http://localhost:${backendPort}/offer`;
    const gcsServerUrl = `ws://localhost:${backendPort}/ws/gcs`;

    const [isWebRTCStreaming, setIsWebRTCStreaming] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isTracking, setIsTracking] = useState(false);
    const [connectionState, setConnectionState] = useState<string>('disconnected');

    const wsRef = useRef<WebSocket | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const pcRef = useRef<RTCPeerConnection | null>(null);

    // High-performance mouse tracking using refs instead of state
    const lastMouseMoveTimeRef = useRef<number>(0);
    const pendingMouseMoveRef = useRef<{ x: number; y: number } | null>(null);
    const MOUSE_THROTTLE_MS = 80;  // throttle to 80ms for better performance

    // WebRTC connection
    const startWebRTC = useCallback(async () => {
        try {
            setConnectionState('connecting');

            // Create peer connection
            // STUN server used by WebRTC to discover the client's public IP/port
            // Used ICE so peers can establish a direct connection through NATs/firewalls
            const pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            pcRef.current = pc;

            // Handle connection state changes
            let statsInterval: ReturnType<typeof setInterval> | null = null;
            pc.onconnectionstatechange = () => {
                setConnectionState(pc.connectionState);
                if (pc.connectionState === 'connected') {
                    setIsWebRTCStreaming(true);
                    setError(null);

                    statsInterval = setInterval(async () => {
                        try {
                            const stats = await pc.getStats();
                            stats.forEach((report) => {
                                if (report.type === 'inbound-rtp' && report.kind === 'video') {
                                    console.log(JSON.stringify({
                                        timestamp_ms: report.timestamp,
                                        framesReceived: report.framesReceived ?? null,
                                        framesDropped: report.framesDropped ?? null,
                                        framesPerSecond: report.framesPerSecond ?? null,
                                    }));
                                }
                            });
                        } catch (_) {}
                    }, 1000);

                } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
                    if (statsInterval) { clearInterval(statsInterval); statsInterval = null; }
                    setIsWebRTCStreaming(false);
                    // Attempt reconnection after a delay
                    setTimeout(startWebRTC, 3000);
                }
            };

            // Handle incoming tracks
            pc.ontrack = (event) => {
                if (videoRef.current && event.streams[0]) {
                    videoRef.current.srcObject = event.streams[0];
                    videoRef.current.play().then(() => {
                        setIsWebRTCStreaming(true);
                    }).catch((err) => {
                        console.error('Video play failed:', err);
                        setIsWebRTCStreaming(true);
                    });
                }
            };

            // Add transceiver to receive video
            pc.addTransceiver('video', { direction: 'recvonly' });

            // Create and set local description (offer)
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Wait for ICE gathering to complete
            await new Promise<void>((resolve) => {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const checkState = () => {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });

            // Send offer to server and get answer
            const response = await fetch(webrtcUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: pc.localDescription?.sdp,
                    type: pc.localDescription?.type
                })
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            const answer = await response.json();

            // Set remote description (answer)
            await pc.setRemoteDescription(new RTCSessionDescription(answer));

        } catch (err) {
            console.error('WebRTC connection error:', err);
            setError('Failed to connect to video stream. Is the AI processor running?');
            setIsWebRTCStreaming(false);
            setConnectionState('failed');
            // Retry connection
            setTimeout(startWebRTC, 3000);
        }
    }, [webrtcUrl]);

    // Initialize WebRTC on mount
    useEffect(() => {
        startWebRTC();

        return () => {
            if (pcRef.current) {
                pcRef.current.close();
                pcRef.current = null;
            }
        };
    }, [startWebRTC]);

    // WebSocket connection for commands
    useEffect(() => {
        const connectWebSocket = () => {
            try {
                const ws = new WebSocket(gcsServerUrl);

                ws.onopen = () => {
                    console.log('WebSocket connected');
                    setConnectionState('connected');
                };

                ws.onclose = () => {
                    console.log('WebSocket disconnected, reconnecting...');
                    setTimeout(connectWebSocket, 3000);
                };

                ws.onerror = (error) => {
                    setError('WebSocket connection error');
                };

                wsRef.current = ws;
            } catch (error) {
                console.error('Failed to connect WebSocket:', error);
                setTimeout(connectWebSocket, 3000);
            }
        };

        connectWebSocket();

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [gcsServerUrl]); // Dependency array, only rerun if the url changes...

    const handleMouseMove = useCallback((e: React.MouseEvent<HTMLVideoElement>) => {
        if (!videoRef.current) return;

        const now = Date.now();
        if (now - lastMouseMoveTimeRef.current < MOUSE_THROTTLE_MS) {
            // Still update pending position even if we don't send yet
            const rect = videoRef.current.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const y = Math.round(e.clientY - rect.top);
            const scaleX = videoRef.current.videoWidth / rect.width;
            const scaleY = videoRef.current.videoHeight / rect.height;
            pendingMouseMoveRef.current = {
                x: Math.round(x * scaleX),
                y: Math.round(y * scaleY)
            };
            return;
        }

        lastMouseMoveTimeRef.current = now;

        const rect = videoRef.current.getBoundingClientRect();
        const x = Math.round(e.clientX - rect.left);
        const y = Math.round(e.clientY - rect.top);
        const scaleX = videoRef.current.videoWidth / rect.width;
        const scaleY = videoRef.current.videoHeight / rect.height;
        const actualX = Math.round(x * scaleX);
        const actualY = Math.round(y * scaleY);

        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'mouse_move',
                x: actualX,
                y: actualY
            }));
        }
    }, []);

    // Send click event to backend
    const handleClick = useCallback((e: React.MouseEvent<HTMLVideoElement>) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        if (!videoRef.current) return;

        const rect = videoRef.current.getBoundingClientRect();
        const x = Math.round(e.clientX - rect.left);
        const y = Math.round(e.clientY - rect.top);

        // Scale coordinates to actual video dimensions
        const scaleX = videoRef.current.videoWidth / rect.width;
        const scaleY = videoRef.current.videoHeight / rect.height;
        const actualX = Math.round(x * scaleX);
        const actualY = Math.round(y * scaleY);

        wsRef.current.send(JSON.stringify({
            type: 'click',
            x: actualX,
            y: actualY
        }));

        setIsTracking(true);
    }, []);

    return (
        <div className="w-full h-full relative bg-gray-900">
            <div ref={containerRef} className="relative w-full h-full bg-gray-800 overflow-hidden">
                {error ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center w-full h-full text-red-400 p-8 text-center bg-gray-900/90">
                        <svg id='error-svg' className="w-12 h-12 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"></path>
                        </svg>
                        <p className="font-semibold">{error}</p>
                        <p className="text-sm mt-2 text-gray-400">Ensure AI processor is running with WebRTC enabled.</p>
                        <p className="text-xs mt-1 text-gray-500">Connection state: {connectionState}</p>
                    </div>
                ) : (
                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className={`absolute inset-0 w-full h-full object-cover cursor-crosshair ${isWebRTCStreaming ? 'opacity-100' : 'opacity-50'}`}
                        onError={() => {
                            setIsWebRTCStreaming(false);
                            setError("Stream error. Is the Python server running?");
                        }}                        
                        onMouseMove={handleMouseMove}
                        onClick={handleClick}
                    />
                )}

                {!isWebRTCStreaming && !error && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 backdrop-blur-sm z-10">
                        <svg className="animate-spin h-8 w-8 text-indigo-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span className="ml-3 text-white">Connecting to video stream...</span>
                        <span className="ml-2 text-xs text-gray-400">({connectionState})</span>
                    </div>
                )}
            </div>
        </div>
    );
}
