"""
Data Loader for TCN-GNN Model

Handles loading and preprocessing of:
1. Hourly sequence features (time-series, last 24-48 hours)
2. Daily satellite features (once per day)
3. Static site features (latitude, longitude)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AirQualityDataset(Dataset):
    """
    Dataset for air quality prediction with TCN-GNN model
    
    Handles:
    - Hourly sequence features (24-48 hour windows)
    - Daily satellite features (extracted once per day)
    - Static site features (latitude, longitude)
    """
    
    def __init__(
        self,
        data_dir: Path,
        site_ids: List[int],
        seq_length: int = 24,
        prediction_horizon: int = 1,
        mode: str = 'train',
        scaler_hourly: StandardScaler = None,
        scaler_daily: StandardScaler = None,
        scaler_static: StandardScaler = None
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            site_ids: List of site IDs to include (e.g., [1, 2, 3, 4, 5, 6, 7])
            seq_length: Length of hourly sequence window (default 24 hours)
            prediction_horizon: Hours ahead to predict (default 1)
            mode: 'train' or 'test'
            scaler_hourly: Pre-fitted scaler for hourly features (if None, fit new)
            scaler_daily: Pre-fitted scaler for daily features (if None, fit new)
            scaler_static: Pre-fitted scaler for static features (if None, fit new)
        """
        self.data_dir = Path(data_dir)
        self.site_ids = site_ids
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        
        # Site coordinates (latitude, longitude)
        self.site_coords = {
            1: (28.69536, 77.18168),
            2: (28.5718, 77.07125),
            3: (28.58278, 77.23441),
            4: (28.82286, 77.10197),
            5: (28.53077, 77.27123),
            6: (28.72954, 77.09601),
            7: (28.71052, 77.24951),
        }
        
        # Define feature groups
        self.hourly_features = [
            'NO2_forecast', 'O3_forecast', 'T_forecast',
            'wind_speed', 'wind_dir_deg', 'blh_forecast',
            'sin_hour', 'cos_hour',
            'NO2_forecast_per_blh', 'O3_forecast_per_blh',
            'blh_delta_1h', 'blh_rel_change',
            'cosSZA', 'solar_elevation', 'sunset_flag'
        ]
        
        self.daily_satellite_features = [
            'NO2_satellite_filled',
            'HCHO_satellite_filled',
            'ratio_satellite'
        ]
        
        self.static_features = ['latitude', 'longitude']
        
        # Load and preprocess data
        self.data = self._load_data()
        self.samples = self._create_samples()
        
        # Fit or use provided scalers
        if scaler_hourly is None:
            self.scaler_hourly = StandardScaler()
            self.scaler_daily = StandardScaler()
            self.scaler_static = StandardScaler()
            self._fit_scalers()
        else:
            self.scaler_hourly = scaler_hourly
            self.scaler_daily = scaler_daily
            self.scaler_static = scaler_static
    
    def _load_data(self) -> Dict[int, pd.DataFrame]:
        """Load data for all sites"""
        data = {}
        for site_id in self.site_ids:
            if self.mode == 'train':
                filename = f'site_{site_id}_train_data_with_satellite.csv'
            else:
                filename = f'site_{site_id}_unseen_input_data_with_satellite.csv'
            
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            
            # Sort by date/time
            df = df.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
            
            # Add static features
            lat, lon = self.site_coords[site_id]
            df['latitude'] = lat
            df['longitude'] = lon
            
            # Ensure all required columns exist
            missing_cols = set(self.hourly_features + self.daily_satellite_features) - set(df.columns)
            if missing_cols:
                print(f"Warning: Missing columns in site {site_id}: {missing_cols}")
                # Fill missing columns with zeros
                for col in missing_cols:
                    df[col] = 0.0
            
            data[site_id] = df
        
        return data
    
    def _create_samples(self) -> List[Dict]:
        """Create samples with sequences (handles cross-day sequences)"""
        samples = []
        
        for site_id, df in self.data.items():
            # Sort by date/time
            df = df.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(int))
            
            # Extract static features once per site (they're the same for all rows)
            # Get from site coordinates directly to avoid pandas memory issues
            lat, lon = self.site_coords[site_id]
            static = np.array([lat, lon])
            
            # Create sequences across the entire dataset
            for i in range(len(df)):
                # Skip if not enough history
                if i < self.seq_length - 1:
                    continue
                
                # Get sequence window (may span multiple days)
                start_idx = i - self.seq_length + 1
                seq_data = df.iloc[start_idx:i+1].copy()
                
                # Extract daily satellite features (use current day's values)
                # CRITICAL: Use unique per day, only first valid entry
                current_date = df.iloc[i]['date']
                current_day_data = df[df['date'] == current_date]
                
                if len(current_day_data) > 0:
                    # Drop duplicates and get first valid (non-null) entry
                    daily_satellite_df = current_day_data[self.daily_satellite_features].drop_duplicates()
                    if len(daily_satellite_df) > 0:
                        # Get first non-null row, fallback to first row if all are null
                        non_null_df = daily_satellite_df.dropna()
                        if len(non_null_df) > 0:
                            daily_satellite = non_null_df.iloc[0].values
                        else:
                            # All values are null, use first row anyway (will be handled by scaler)
                            daily_satellite = daily_satellite_df.iloc[0].values
                    else:
                        # Fallback: use last available daily satellite values from sequence
                        daily_satellite = seq_data[self.daily_satellite_features].drop_duplicates().iloc[-1].values if len(seq_data) > 0 else np.zeros(len(self.daily_satellite_features))
                else:
                    # Fallback: use last available daily satellite values from sequence
                    daily_satellite = seq_data[self.daily_satellite_features].drop_duplicates().iloc[-1].values if len(seq_data) > 0 else np.zeros(len(self.daily_satellite_features))
                
                # Static features already extracted above (same for all rows in site)
                # No need to extract again per row
                
                # Extract hourly sequence
                hourly_seq = seq_data[self.hourly_features].values
                
                # Ensure sequence length matches (should always be seq_length)
                if len(hourly_seq) < self.seq_length:
                    # Pad with zeros if needed (shouldn't happen, but safety check)
                    padding = np.zeros((self.seq_length - len(hourly_seq), len(self.hourly_features)))
                    hourly_seq = np.vstack([padding, hourly_seq])
                
                # Get target (if available) - CRITICAL: Skip if prediction horizon goes beyond dataset
                if 'NO2_target' in df.columns and 'O3_target' in df.columns:
                    target_idx = i + self.prediction_horizon
                    if target_idx >= len(df):
                        # Skip this sample - prediction horizon goes beyond dataset
                        continue
                    target = df[['NO2_target', 'O3_target']].iloc[target_idx].values
                    # Skip if target contains NaN
                    if np.isnan(target).any():
                        continue
                else:
                    # Skip samples without targets (test mode)
                    continue
                
                samples.append({
                    'site_id': site_id,
                    'hourly_seq': hourly_seq,
                    'daily_satellite': daily_satellite,
                    'static': static,
                    'target': target,
                    'date': current_date,
                    'hour': df.iloc[i]['hour']
                })
        
        return samples
    
    def _fit_scalers(self):
        """
        Fit scalers on training data.
        
        For very large datasets, this could use significant memory.
        Consider fitting on a sample subset if memory is constrained.
        """
        hourly_data = []
        daily_data = []
        static_data = []
        
        # For memory efficiency, could sample subset for very large datasets
        # For now, use all data (works fine for moderate datasets)
        for sample in self.samples:
            hourly_data.append(sample['hourly_seq'])
            daily_data.append(sample['daily_satellite'])
            static_data.append(sample['static'])
        
        # Stack arrays (could be memory-intensive for very large datasets)
        hourly_data = np.vstack(hourly_data)
        daily_data = np.vstack(daily_data)
        static_data = np.vstack(static_data)
        
        # Fit scalers
        self.scaler_hourly.fit(hourly_data.reshape(-1, hourly_data.shape[-1]))
        self.scaler_daily.fit(daily_data)
        self.scaler_static.fit(static_data)
        
        # Clear large arrays from memory
        del hourly_data, daily_data, static_data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Scale features
        hourly_seq = sample['hourly_seq']
        hourly_seq_scaled = self.scaler_hourly.transform(
            hourly_seq.reshape(-1, hourly_seq.shape[-1])
        ).reshape(hourly_seq.shape)
        
        daily_satellite_scaled = self.scaler_daily.transform(
            sample['daily_satellite'].reshape(1, -1)
        ).flatten()
        
        static_scaled = self.scaler_static.transform(
            sample['static'].reshape(1, -1)
        ).flatten()
        
        # Convert to tensors
        hourly_seq_tensor = torch.FloatTensor(hourly_seq_scaled)
        daily_satellite_tensor = torch.FloatTensor(daily_satellite_scaled)
        static_tensor = torch.FloatTensor(static_scaled)
        
        # Target
        if not np.isnan(sample['target']).any():
            target_tensor = torch.FloatTensor(sample['target'])
        else:
            target_tensor = torch.FloatTensor([0.0, 0.0])  # Placeholder for test data
        
        return {
            'hourly_seq': hourly_seq_tensor,
            'daily_satellite': daily_satellite_tensor,
            'static': static_tensor,
            'target': target_tensor,
            'site_id': sample['site_id']
        }


# Global graph edges based on site spatial relationships
# This is created once and reused for all batches
_global_edge_index = None
_global_edge_distances = None  # Store distances for potential edge attributes
_site_coords = {
    1: (28.69536, 77.18168),
    2: (28.5718, 77.07125),
    3: (28.58278, 77.23441),
    4: (28.82286, 77.10197),
    5: (28.53077, 77.27123),
    6: (28.72954, 77.09601),
    7: (28.71052, 77.24951),
}


def haversine_matrix(coords_array):
    """
    Compute pairwise Haversine distances between coordinates.
    
    Args:
        coords_array: [[lat, lon], ...] in degrees
        
    Returns:
        distance_matrix: [n_sites, n_sites] distance matrix in kilometers
    """
    R = 6371.0  # Earth radius in km
    lat = np.radians(coords_array[:, 0])[:, None]
    lon = np.radians(coords_array[:, 1])[:, None]
    
    dlat = lat - lat.T
    dlon = lon - lon.T
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    return d


def _get_global_edge_index(k_neighbors=3):
    """
    Create global graph edges based on site spatial relationships.
    This creates edges between sites based on physical distance (K-nearest neighbors).
    
    Uses Haversine distance (geodesic) for accurate distance calculation on Earth's surface.
    """
    global _global_edge_index
    if _global_edge_index is None:
        site_ids = sorted(_site_coords.keys())  # Consistent ordering is critical!
        coords = [(_site_coords[sid][0], _site_coords[sid][1]) for sid in site_ids]
        coords_array = np.array(coords)
        
        # Compute pairwise distances using Haversine (geodesic) distance
        # This is more accurate than Euclidean for lat/lon coordinates
        distances = haversine_matrix(coords_array)
        
        # For each site, find k nearest neighbors (excluding self)
        # Use dictionary to track unique edges and their distances
        unique_edge_dict = {}
        
        for i in range(len(site_ids)):
            # Get k+1 nearest (including self), then exclude self
            nearest = np.argsort(distances[i])[:k_neighbors + 1]
            nearest = nearest[nearest != i]  # Remove self
            
            for j in nearest:
                # Use sorted tuple as key to ensure uniqueness (undirected edge)
                edge_key = tuple(sorted([i, j]))
                if edge_key not in unique_edge_dict:
                    unique_edge_dict[edge_key] = distances[i, j]
        
        # Convert unique edges to list (both directions for undirected graph)
        # This ensures GAT can traverse edges in both directions
        unique_edges = []
        edge_distances_list = []
        for edge_key, dist in unique_edge_dict.items():
            i, j = edge_key
            unique_edges.append([i, j])
            unique_edges.append([j, i])  # Undirected: add both directions
            edge_distances_list.extend([dist, dist])  # Same distance for both directions
        
        if len(unique_edges) > 0:
            _global_edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
            _global_edge_distances = np.array(edge_distances_list)
        else:
            _global_edge_index = torch.empty((2, 0), dtype=torch.long)
            _global_edge_distances = np.array([])
    
    return _global_edge_index


def collate_fn(batch):
    """
    Custom collate function for batching samples
    
    Handles batching of sequences and creates graph structure.
    Uses GLOBAL spatial graph edges based on site locations (K-nearest neighbors).
    This preserves spatial relationships between actual sites.
    """
    hourly_seqs = [item['hourly_seq'] for item in batch]
    daily_satellites = [item['daily_satellite'] for item in batch]
    statics = [item['static'] for item in batch]
    targets = [item['target'] for item in batch]
    site_ids = [item['site_id'] for item in batch]
    
    # Stack tensors
    hourly_seqs = torch.stack(hourly_seqs)
    daily_satellites = torch.stack(daily_satellites)
    statics = torch.stack(statics)
    targets = torch.stack(targets)
    
    # Use GLOBAL graph edges based on spatial relationships between sites
    # This creates edges between sites based on physical distance (K-nearest neighbors)
    # NOT fully connected within batch - preserves actual spatial structure
    global_edge_index = _get_global_edge_index(k_neighbors=3)
    
    # Map global site indices to batch indices
    # CRITICAL: site_id_list must match the ordering used in _get_global_edge_index
    # Using sorted(_site_coords.keys()) ensures consistent ordering
    site_id_list = sorted(_site_coords.keys())
    
    # Defensive check: ensure all sites in batch are known
    for site_id in site_ids:
        if site_id not in site_id_list:
            raise KeyError(f"Unknown site id in batch: {site_id}. Known sites: {site_id_list}")
    
    site_to_batch_indices = {}
    for batch_idx, site_id in enumerate(site_ids):
        if site_id not in site_to_batch_indices:
            site_to_batch_indices[site_id] = []
        site_to_batch_indices[site_id].append(batch_idx)
    
    # Create batch-specific edges based on global graph structure
    batch_edges = []
    
    # For each edge in global graph, create corresponding edges in batch
    if global_edge_index.shape[1] > 0:
        for edge_idx in range(global_edge_index.shape[1]):
            i_global, j_global = global_edge_index[:, edge_idx].cpu().numpy().tolist()
            site_i = site_id_list[i_global]
            site_j = site_id_list[j_global]
            
            # Create edges between all batch samples from site_i to site_j
            if site_i in site_to_batch_indices and site_j in site_to_batch_indices:
                for batch_i in site_to_batch_indices[site_i]:
                    for batch_j in site_to_batch_indices[site_j]:
                        # Connect nodes from different sites (spatial relationship)
                        # Also allow self-connections if same site but different time steps
                        batch_edges.append([batch_i, batch_j])
    
    if len(batch_edges) > 0:
        edge_index = torch.tensor(batch_edges, dtype=torch.long).t().contiguous()
    else:
        # Fallback: if no spatial edges (shouldn't happen with proper site data)
        # Create minimal connectivity to prevent isolated nodes
        n_nodes = len(batch)
        if n_nodes > 1:
            # Create a simple chain to ensure connectivity
            edge_index = torch.tensor([[i for i in range(n_nodes-1)], 
                                       [i+1 for i in range(n_nodes-1)]], dtype=torch.long)
            # Make undirected
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return {
        'hourly_seq': hourly_seqs,
        'daily_satellite': daily_satellites,
        'static': statics,
        'target': targets,
        'edge_index': edge_index,
        'site_ids': site_ids
    }


def create_data_loaders(
    data_dir: Path,
    site_ids: List[int],
    seq_length: int = 24,
    prediction_horizon: int = 1,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    use_time_split: bool = True
):
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Directory containing CSV files
        site_ids: List of site IDs
        seq_length: Length of sequence window
        prediction_horizon: Hours ahead to predict
        batch_size: Batch size
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        use_time_split: If True, split by time (blocked time-series split)
    """
    # Create full dataset
    full_dataset = AirQualityDataset(
        data_dir=data_dir,
        site_ids=site_ids,
        seq_length=seq_length,
        prediction_horizon=prediction_horizon,
        mode='train'
    )
    
    # Split dataset
    n_samples = len(full_dataset)
    
    if use_time_split:
        # Blocked time-series split: train on older dates, validate on newer dates
        # Sort samples by date
        sorted_indices = sorted(
            range(n_samples),
            key=lambda i: (full_dataset.samples[i]['date'], full_dataset.samples[i]['hour'])
        )
        
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        train_indices = sorted_indices[:train_end]
        val_indices = sorted_indices[train_end:val_end]
        test_indices = sorted_indices[val_end:]
    else:
        # Random split
        indices = np.random.permutation(n_samples)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        train_indices = indices[:train_end].tolist()
        val_indices = indices[train_end:val_end].tolist()
        test_indices = indices[val_end:].tolist()
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Optimize data loading for GPU if available
    pin_memory = torch.cuda.is_available()
    num_workers = 4 if torch.cuda.is_available() else 0  # Use multiple workers for GPU
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader, full_dataset.scaler_hourly, full_dataset.scaler_daily, full_dataset.scaler_static
