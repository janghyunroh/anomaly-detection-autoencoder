import os
import pandas as pd
import numpy as np
from utils.config import CONFIG, REQUIRED_FEATURES, PER_MODEL

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Literal, Optional
import warnings
from pathlib import Path

import plotly.io as pio

def visualize_data(
    rpm_type: Literal['rpm_1200', 'rpm_600'],
    model_type: Literal['vibration', 'temperature', 'voltage', 'rpm'],
    data_type: Literal['raw', 'cleaned', 'anomalous'],
    downsample_rate: int = 10
) -> go.Figure:
    """
    Visualize sensor data based on RPM type, model type, and data type.
    
    Args:
        rpm_type: Type of RPM data ('rpm_1200' or 'rpm_600')
        model_type: Type of model ('vibration', 'temperature', 'voltage', 'rpm')
        data_type: Type of data to visualize ('raw', 'cleaned', 'anomalous')
        downsample_rate: Rate at which to downsample the data (default: 10)
    
    Returns:
        plotly.graph_objects.Figure: Interactive visualization
    """
    
    def load_data(rpm_config: dict, data_type: str, model_type: str) -> pd.DataFrame:
        """Load and combine data from specified directory."""
        dfs = []
        
        if data_type == 'raw':
            # Load from train directory
            train_path = Path(rpm_config['raw_data_train_dir'])
            if train_path.exists():
                for file in train_path.glob('*.csv'):
                    df = pd.read_csv(file)
                    df['source'] = f"train_{file.stem}"
                    dfs.append(df)
            
            # Load from test directory
            test_path = Path(rpm_config['raw_data_test_dir'])
            if test_path.exists():
                for file in test_path.glob('*.csv'):
                    df = pd.read_csv(file)
                    df['source'] = f"test_{file.stem}"
                    dfs.append(df)
                    
        elif data_type == 'cleaned':
            # Load from processed train directory
            train_path = Path(rpm_config['processed_train_data_dir'])
            if train_path.exists():
                for file in train_path.glob('*.csv'):
                    df = pd.read_csv(file)
                    df['source'] = f"train_{file.stem}"
                    dfs.append(df)
            
            # Load from processed test directory
            test_path = Path(rpm_config['processed_test_data_dir'])
            if test_path.exists():
                for file in test_path.glob('*.csv'):
                    df = pd.read_csv(file)
                    df['source'] = f"test_{file.stem}"
                    dfs.append(df)
                    
        else:  # anomalous data
            path = Path(rpm_config['anomalous_data_dir'])
            if path.exists():
                # rpm_type에 따라 실제 파일명의 rpm 값 결정
                rpm_value = '600' if rpm_type == 'rpm_600' else '1200'
                
                # 수정된 파일 패턴으로 검색
                file_pattern = f"anomalous_{model_type}_cleaned_{rpm_value}-*.csv"
                matching_files = list(path.glob(file_pattern))
                
                if not matching_files:
                    raise FileNotFoundError(
                        f"No anomalous data files found matching pattern: {file_pattern}"
                    )
                
                for file in matching_files:
                    df = pd.read_csv(file)
                    df['source'] = file.stem
                    dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No data files found for {data_type} data")
            
        return pd.concat(dfs, ignore_index=True)
    
    def optimize_visualization(df: pd.DataFrame, downsample_rate: int) -> pd.DataFrame:
        """Optimize data for visualization based on downsample rate."""
        if downsample_rate == 0:
            # If downsample_rate is 0, use dynamic downsampling based on data size
            total_points = len(df)
            if total_points > 100000:
                downsample_rate = total_points // 100000
                warnings.warn(f"Large dataset detected. Automatically downsampling by factor of {downsample_rate}")
                return df.iloc[::downsample_rate].copy()
            return df
        return df.iloc[::downsample_rate].copy()
    
    try:
        # Get configuration for specified rpm type
        if rpm_type not in CONFIG:
            raise ValueError(f"Invalid rpm_type: {rpm_type}")
        rpm_config = CONFIG[rpm_type]
        
        # Validate model type
        if model_type not in CONFIG['model_type']:
            raise ValueError(f"Invalid model_type: {model_type}")
            
        # Get required features for the model type
        required_features = REQUIRED_FEATURES.get(model_type)
        if not required_features:
            raise ValueError(f"No required features found for model_type: {model_type}")
        
        # Get model-specific configuration
        model_config = PER_MODEL.get(model_type)
        if not model_config:
            raise ValueError(f"No configuration found for model_type: {model_type}")
        
        # Load and preprocess data
        data = load_data(rpm_config, data_type, model_type)
        data = optimize_visualization(data, downsample_rate)
        
        # Create visualization
        if data_type == 'anomalous':
            if model_type == 'vibration':
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  subplot_titles=['X-Axis Acceleration', 'Y-Axis Acceleration', 'Z-Axis Acceleration'])
                
                for idx, col in enumerate(['accel_x', 'accel_y', 'accel_z'], 1):
                    for source in data['source'].unique():
                        source_data = data[data['source'] == source]
                        
                        # Normal data points
                        normal_mask = source_data['label'] == 0
                        fig.add_trace(
                            go.Scatter(
                                x=source_data[normal_mask].index,
                                y=source_data[normal_mask][col],
                                name=f"{source} - {col} (Normal)",
                                mode='lines',
                                line=dict(color='blue')
                            ),
                            row=idx, col=1
                        )
                        
                        # Anomalous data points
                        anomaly_mask = source_data['label'] == 1
                        fig.add_trace(
                            go.Scatter(
                                x=source_data[anomaly_mask].index,
                                y=source_data[anomaly_mask][col],
                                name=f"{source} - {col} (Anomaly)",
                                mode='markers',
                                marker=dict(color='red', size=8)
                            ),
                            row=idx, col=1
                        )
                    
                    fig.update_yaxes(title_text='Acceleration (g)', row=idx, col=1)
            
            else:
                # Single plot for temperature, voltage, or rpm
                fig = go.Figure()
                feature = required_features if isinstance(required_features, str) else required_features[0]
                
                for source in data['source'].unique():
                    source_data = data[data['source'] == source]
                    
                    # Normal data points
                    normal_mask = source_data['label'] == 0
                    fig.add_trace(
                        go.Scatter(
                            x=source_data[normal_mask].index,
                            y=source_data[normal_mask][feature],
                            name=f"{source} - Normal",
                            mode='lines',
                            line=dict(color='blue')
                        )
                    )
                    
                    # Anomalous data points
                    anomaly_mask = source_data['label'] == 1
                    fig.add_trace(
                        go.Scatter(
                            x=source_data[anomaly_mask].index,
                            y=source_data[anomaly_mask][feature],
                            name=f"{source} - Anomaly",
                            mode='markers',
                            marker=dict(color='red', size=8)
                        )
                    )
                
                # Set appropriate y-axis title based on model type
                y_axis_titles = {
                    'temperature': 'Temperature (°C)',
                    'voltage': 'Voltage (V)',
                    'rpm': 'RPM'
                }
                fig.update_yaxes(title_text=y_axis_titles[model_type])
        
        else:
            # Original visualization for non-anomalous data
            if model_type == 'vibration':
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                  subplot_titles=['X-Axis Acceleration', 'Y-Axis Acceleration', 'Z-Axis Acceleration'])
                
                for idx, col in enumerate(['accel_x', 'accel_y', 'accel_z'], 1):
                    for source in data['source'].unique():
                        source_data = data[data['source'] == source]
                        fig.add_trace(
                            go.Scatter(x=source_data.index, y=source_data[col],
                                     name=f"{source} - {col}",
                                     mode='lines'),
                            row=idx, col=1
                        )
                    fig.update_yaxes(title_text='Acceleration (g)', row=idx, col=1)
            else:
                fig = go.Figure()
                feature = required_features if isinstance(required_features, str) else required_features[0]
                
                for source in data['source'].unique():
                    source_data = data[data['source'] == source]
                    fig.add_trace(
                        go.Scatter(x=source_data.index, y=source_data[feature],
                                 name=source,
                                 mode='lines')
                    )
                
                y_axis_titles = {
                    'temperature': 'Temperature (°C)',
                    'voltage': 'Voltage (V)',
                    'rpm': 'RPM'
                }
                fig.update_yaxes(title_text=y_axis_titles[model_type])
        
        # Update layout
        fig.update_layout(
            title=f"{rpm_type} - {model_type} Data ({data_type})",
            xaxis_title="Sample Index",
            height=800 if model_type == 'vibration' else 500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            ),
            width=1000
        )
        
        return fig
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise