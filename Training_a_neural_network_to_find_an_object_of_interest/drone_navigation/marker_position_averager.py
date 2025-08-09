from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MarkerStats:
    """Data class to store marker statistics and samples."""
    samples: list[Tuple[float, float]]
    sum_x: float
    sum_y: float
    avg_coords: Optional[Tuple[float, float]]


class MarkerPositionAverager:
    """Implements a moving average filter for marker positions using sliding window approach.
    
    This class provides robust position estimation by averaging multiple samples,
    reducing noise and improving tracking stability.
    
    Attributes:
        min_samples (int): Minimum samples required before averaging
        max_samples (int): Maximum samples to store per marker
        marker_data (Dict[int, MarkerStats]): Stores tracking data for each marker
        
    Example:
        >>> averager = MarkerPositionAverager(min_samples=5, max_samples=20)
        >>> averager.add_marker_sample({1: (10.5, 15.3), 2: (20.1, 25.7)})
        >>> avg_coords = averager.get_marker_coords_by_id(1)
    """
    
    def __init__(self, min_samples: int = 10, max_samples: int = 100) -> None:
        """Initialize the averager with configuration parameters.
        
        Args:
            min_samples: Minimum samples required before returning averaged coordinates
            max_samples: Maximum samples to maintain in the sliding window
            
        Raises:
            ValueError: If min_samples > max_samples or either is <= 0
        """

        if min_samples <= 0 or max_samples <= 0:
            raise ValueError("Sample counts must be positive integers")
        if min_samples > max_samples:
            raise ValueError("min_samples cannot exceed max_samples")
            

        self.marker_data: Dict[int, MarkerStats] = {}
        self.min_samples = min_samples
        self.max_samples = max_samples
        logger.info(f"Initialized MarkerPositionAverager with min_samples={min_samples}, max_samples={max_samples}")


    def add_marker_sample(self, markers_dict: Dict[int, Optional[Tuple[float, float]]]) -> None:
        """Add new marker coordinates to the averaging buffer.
        
        Args:
            markers_dict: Dictionary mapping marker IDs to their coordinates (x, y)
                          or None if marker was not detected
                          
        Note:
            Coordinates are only added if both x and y are valid numbers
        """
        if not markers_dict:
            logger.debug("Received empty markers dict")
            return
            
        for marker_id, coords in markers_dict.items():
            if coords is None:
                logger.debug(f"Skipping None coordinates for marker {marker_id}")
                continue
                
            self._process_marker_sample(marker_id, coords)


    def _process_marker_sample(self, marker_id: int, coords: Tuple[float, float]) -> None:
        """Internal method to process individual marker samples."""
        if marker_id not in self.marker_data:
            self.marker_data[marker_id] = MarkerStats(
                samples=[],
                sum_x=0.0,
                sum_y=0.0,
                avg_coords=None
            )
            
        data = self.marker_data[marker_id]
        data.samples.append(coords)
        data.sum_x += coords[0]
        data.sum_y += coords[1]
        
        # Maintain sliding window size
        if len(data.samples) > self.max_samples:
            old_x, old_y = data.samples.pop(0)
            data.sum_x -= old_x
            data.sum_y -= old_y
            
        # Update average if enough samples
        if len(data.samples) >= self.min_samples:
            avg_x = round(data.sum_x / len(data.samples), 2)
            avg_y = round(data.sum_y / len(data.samples), 2)
            data.avg_coords = (avg_x, avg_y)
            logger.debug(f"Updated average for marker {marker_id}: {data.avg_coords}")


    def get_marker_coords_by_id(self, marker_id: int) -> Optional[Tuple[float, float]]:
        """Get averaged coordinates for a specific marker.
        
        Args:
            marker_id: ID of the marker to query
            
        Returns:
            Tuple of (x, y) coordinates if available and enough samples collected,
            None otherwise
        """
        marker = self.marker_data.get(marker_id)
        if marker and marker.avg_coords:
            return marker.avg_coords
        return None


    def get_all_markers_coords(self) -> Dict[int, Tuple[float, float]]:
        """Get averaged coordinates for all markers with sufficient samples.
        
        Returns:
            Dictionary mapping marker IDs to their averaged coordinates
            Only includes markers that have min_samples or more samples
        """
        return {
            marker_id: data.avg_coords
            for marker_id, data in self.marker_data.items()
            if data.avg_coords is not None
        }

    def reset_marker(self, marker_id: int) -> None:
        """Reset tracking data for a specific marker.
        
        Args:
            marker_id: ID of the marker to reset
        """
        if marker_id in self.marker_data:
            del self.marker_data[marker_id]
            logger.info(f"Reset tracking data for marker {marker_id}")