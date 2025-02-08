import hashlib
import json
from pathlib import Path
import numpy as np
from typing import Optional, Tuple
from deep_orderbook.config import ShaperConfig

class ArrayCache:
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_config_hash(self, shaper_config: ShaperConfig) -> str:
        """Create a unique hash for the shaper config parameters that affect array generation"""
        relevant_params = {
            'zoom_frac': shaper_config.zoom_frac,
            'num_side_lvl': shaper_config.num_side_lvl,
            'rolling_window_size': shaper_config.rolling_window_size,
            'look_ahead': shaper_config.look_ahead,
            'look_ahead_side_bips': shaper_config.look_ahead_side_bips,
            'look_ahead_side_width': shaper_config.look_ahead_side_width
        }
        config_str = json.dumps(relevant_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, data_file: Path, shaper_config: ShaperConfig) -> Path:
        """Generate cache file path based on input file and config"""
        config_hash = self._get_config_hash(shaper_config)
        cache_name = f"{data_file.stem}_{config_hash}.npz"
        return self.cache_dir / cache_name
    
    def load_cached(self, data_file: Path, shaper_config: ShaperConfig) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load cached arrays if they exist"""
        cache_path = self._get_cache_path(data_file, shaper_config)
        if not cache_path.exists():
            return None
            
        try:
            with np.load(cache_path) as data:
                books_array = data['books_array']
                time_levels = data['time_levels']
                prices_array = data['prices_array']
            return books_array, time_levels, prices_array
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def save_to_cache(self, 
                     data_file: Path,
                     shaper_config: ShaperConfig,
                     books_array: np.ndarray,
                     time_levels: np.ndarray,
                     prices_array: np.ndarray) -> None:
        """Save processed arrays to cache"""
        cache_path = self._get_cache_path(data_file, shaper_config)
        np.savez_compressed(
            cache_path,
            books_array=books_array,
            time_levels=time_levels,
            prices_array=prices_array
        )

    def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear old cache files"""
        if older_than_days is not None:
            import time
            current_time = time.time()
            for cache_file in self.cache_dir.glob("*.npz"):
                if (current_time - cache_file.stat().st_mtime) > (older_than_days * 86400):
                    cache_file.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.npz"):
                cache_file.unlink() 