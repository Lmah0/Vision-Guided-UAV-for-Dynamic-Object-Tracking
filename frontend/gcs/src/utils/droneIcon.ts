import React from 'react';
import SvgIcon, { SvgIconProps } from '@mui/material/SvgIcon';

export function DroneIcon(props: SvgIconProps) {
    return React.createElement(
        SvgIcon,
        { ...props, viewBox: '0 0 24 24' },
        React.createElement('circle', { cx: '6', cy: '6', r: '2.2', fill: 'currentColor' }),
        React.createElement('circle', { cx: '18', cy: '6', r: '2.2', fill: 'currentColor' }),
        React.createElement('circle', { cx: '6', cy: '18', r: '2.2', fill: 'currentColor' }),
        React.createElement('circle', { cx: '18', cy: '18', r: '2.2', fill: 'currentColor' }),
        React.createElement('path', {
            d: 'M8 8 L10.5 10.5 M16 8 L13.5 10.5 M8 16 L10.5 13.5 M16 16 L13.5 13.5',
            stroke: 'currentColor',
            strokeWidth: '1.6',
            strokeLinecap: 'round',
            fill: 'none',
        }),
        React.createElement('rect', { x: '10', y: '10', width: '4', height: '4', rx: '0.8', fill: 'currentColor' })
    );
}