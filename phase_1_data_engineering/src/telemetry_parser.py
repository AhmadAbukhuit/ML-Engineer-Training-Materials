"""
Core Idea: This script acts as the foundational Data Engineering pipeline for telemetry data.
Machine Learning models cannot be trained on messy, fragmented raw data (like nested JSONs and XMLs).
This pipeline performs ETL (Extract, Transform, Load):
1. Extract: Reads raw JSON and XML telemetry logs from IoT devices.
2. Transform: Flattens nested structures, unifies them into a single tabular schema using Polars, 
              removes duplicates, and handles missing values.
3. Load: Exports the cleaned, unified dataset as a CSV file.

The resulting `unified_telemetry.csv` is the "Golden Dataset" that will be fed into our 
Machine Learning models for training in the subsequent phases.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
import polars as pl

def parse_json_log(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts and flattens data from JSON telemetry logs.
    
    ML Context: Nested JSON structures (like a 'gps' dictionary inside a main dictionary) 
    cannot be directly fed into tabular ML models (like XGBoost or Random Forests). 
    This function flattens the hierarchy into a 1D dictionary (row) with explicit features 
    like 'lat', 'lon', and 'temperature'.
    """
    records = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            # Handle both single JSON objects and arrays of objects
            if isinstance(data, dict):
                data = [data]
                
            for entry in data:
                # Flatten the nested JSON structure into a tabular format
                record = {
                    "timestamp": entry.get("timestamp"),
                    "device_id": entry.get("device_id"),
                    "temperature": entry.get("sensor_1", {}).get("temperature"),
                    "lat": entry.get("gps", {}).get("coordinates", {}).get("lat"),
                    "lon": entry.get("gps", {}).get("coordinates", {}).get("lon"),
                }
                records.append(record)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON {file_path.name}: {e}")
        
    return records

def parse_xml_log(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts and flattens data from XML telemetry logs.
    
    ML Context: Similar to JSON, XML data is hierarchical. We extract the relevant 
    text nodes and cast them to floats where appropriate so the ML model receives 
    numerical continuous features instead of raw strings.
    """
    records = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Assume root contains multiple <reading> or <log> tags
        for entry in root.findall('.//reading'):
            record = {
                "timestamp": entry.findtext('timestamp'),
                "device_id": entry.findtext('device_id'),
            }
            
            # Extract nested sensor data
            sensor = entry.find('sensor')
            if sensor is not None:
                record["temperature"] = float(sensor.findtext('temperature')) if sensor.findtext('temperature') else None
                
            # Extract nested GPS data
            gps = entry.find('gps')
            if gps is not None:
                record["lat"] = float(gps.findtext('lat')) if gps.findtext('lat') else None
                record["lon"] = float(gps.findtext('lon')) if gps.findtext('lon') else None
                
            records.append(record)
    except ET.ParseError as e:
        print(f"Error parsing XML {file_path.name}: {e}")
        
    return records

def process_telemetry_directory(raw_dir: str, processed_dir: str) -> None:
    """
    The main ETL pipeline orchestrator.
    
    ML Context: Training data must be perfectly clean. This function unifies data from 
    different sources (JSON/XML) into a single Polars DataFrame, drops invalid rows, 
    removes duplicates (to prevent data leakage or biased training), and formats timestamps.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Ensure the output directory exists
    processed_path.mkdir(parents=True, exist_ok=True)
    
    all_records: List[Dict[str, Any]] = []
    
    # 1. Ingestion Phase: Iterate through files and route to correct parser
    for file_path in raw_path.iterdir():
        if file_path.suffix == '.json':
            all_records.extend(parse_json_log(file_path))
        elif file_path.suffix == '.xml':
            all_records.extend(parse_xml_log(file_path))
            
    if not all_records:
        print("No telemetry records found. Check your raw data directory.")
        return

    # 2. Polars DataFrame Creation
    # Polars is a highly optimized DataFrame library (faster than Pandas).
    # It automatically infers the data schema (types) from our list of dictionaries.
    df = pl.DataFrame(all_records)
    
    print(f"Initial row count: {df.height}")
    
    # 3. Data Cleaning Pipeline (Crucial for ML)
    # Drop rows where the timestamp is missing. ML time-series models cannot use un-timestamped data.
    df = df.drop_nulls(subset=["timestamp"])
    
    # Drop exact duplicates based on timestamp and device_id.
    # Duplicates in training data can cause models to overfit to repeated patterns.
    df = df.unique(subset=["timestamp", "device_id"], maintain_order=True)
    
    # Cast timestamp strings to proper datetime objects.
    # This allows future ML models to extract temporal features (e.g., 'hour_of_day', 'day_of_week').
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False)
    )
    
    print(f"Cleaned row count: {df.height}")
    
    # 4. Export
    # Save the cleaned tabular data as a CSV. This file will be loaded by our ML training scripts.
    output_file = processed_path / "unified_telemetry.csv"
    df.write_csv(output_file)
    print(f"Successfully exported unified ML-ready dataset to {output_file}")

if __name__ == "__main__":
    # Define paths relative to this script's location so it works from any working directory
    # __file__ is src/telemetry_parser.py
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # Go up two levels: src/ -> phase_1/ -> repo_root/
    RAW_DATA_DIR = SCRIPT_DIR.parent.parent / "data/raw/telemetry"
    PROCESSED_DATA_DIR = SCRIPT_DIR.parent.parent / "data/processed"
    
    process_telemetry_directory(RAW_DATA_DIR, PROCESSED_DATA_DIR)