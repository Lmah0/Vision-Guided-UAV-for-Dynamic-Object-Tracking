""" Shared database utilities for GCS backend """
from dotenv import load_dotenv
import boto3
import os
import csv
from decimal import Decimal
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, List
from pathlib import Path

load_dotenv(dotenv_path="../../.env")

# Initialize DynamoDB connection
dynamodb = boto3.resource(
    'dynamodb',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
table = dynamodb.Table(os.getenv('DYNAMODB_TABLE_NAME'))

# CSV storage configuration
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
Path(RECORDINGS_DIR).mkdir(exist_ok=True)

def get_all_objects() -> List[Dict[str, Any]]:
    """Retrieve a list of all recorded objects with their classifications and timestamps"""
    response = table.scan()
    items = response.get('Items', [])
    object_list = []
    
    for item in items:
        # Get the first timestamp from positions array
        first_timestamp = None
        positions = item.get('positions', [])
        if positions and len(positions) > 0:
            first_timestamp = positions[0].get('ts')
        
        object_data = {
            "objectID": item['objectID'], 
            "classification": item['class'],
            "timestamp": first_timestamp
        }
        object_list.append(object_data)
    
    return object_list

def delete_object(object_id: str) -> bool:
    """Delete a recorded object from the DynamoDB table by its ID"""
    try:
        table.delete_item(Key={'objectID': object_id})
        return True
    except Exception as e:
        print(f"Error deleting object {object_id}: {e}")
        return False

def record_telemetry_data(data: List[Dict[str, Any]], classification: str = 'Unknown') -> str:
    """Save recording data to a CSV file. Creates a separate file for each recording.
    
    Args:
        data: List of telemetry points with timestamp, latitude, longitude, altitude, speed, heading
        classification: Object classification/type (used in filename)
    
    Returns:
        Path to the saved CSV file
    """
    if not data or len(data) == 0:
        raise ValueError("No recording data found in message")
    
    # Generate unique filename based on timestamp and classification
    object_id = str(uuid.uuid4())[:8]
    timestamp_str = datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp_str}_{classification}_{object_id}.csv"
    filepath = os.path.join(RECORDINGS_DIR, filename)
    
    # Write to CSV file
    try:
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'latitude', 'longitude', 'altitude', 'speed', 'heading', 'classification']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for point in data:
                writer.writerow({
                    'timestamp': datetime.fromtimestamp(point['timestamp'], tz=timezone.utc).isoformat(),
                    'latitude': point.get('latitude', 0),
                    'longitude': point.get('longitude', 0),
                    'altitude': point.get('altitude', 0),
                    'speed': point.get('speed', 0),
                    'heading': point.get('heading', 0),
                    'classification': classification,
                })
        
        print(f"Telemetry recorded to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving telemetry to CSV: {e}")
        raise
