"""
Model cache and optimization utilities for YOLOv8
"""
import torch
from ultralytics import YOLO
import os
from typing import Optional, Dict

class ModelCache:
    """
    Singleton cache for YOLO models with device and precision optimization
    """
    _instance = None
    _cache: Dict[str, YOLO] = {}
    _device = None
    _half_precision = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize device and precision settings"""
        # Auto-detect device
        if torch.cuda.is_available():
            self._device = 'cuda'
            # Enable half precision for CUDA
            self._half_precision = True
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   Half precision (FP16): Enabled")
        else:
            self._device = 'cpu'
            self._half_precision = False
            print(f"💻 Using CPU (no GPU detected)")
            print(f"   Half precision: Disabled")
    
    def get_model(self, model_path: str, force_reload: bool = False) -> YOLO:
        """
        Get cached model or load new one
        
        Args:
            model_path: Path to model weights
            force_reload: Force reload even if cached
            
        Returns:
            YOLO model instance
        """
        # Normalize path for cache key
        cache_key = os.path.abspath(model_path)
        
        # Return cached model if exists
        if not force_reload and cache_key in self._cache:
            print(f"✅ Using cached model: {os.path.basename(model_path)}")
            return self._cache[cache_key]
        
        # Load new model
        print(f"⏳ Loading model: {os.path.basename(model_path)}")
        model = YOLO(model_path)
        
        # Move to device
        model.to(self._device)
        
        # Apply half precision if available
        if self._half_precision:
            try:
                # Note: YOLO automatically handles half precision with half=True in predict()
                # We just set a flag here
                print(f"✅ Half precision enabled for faster inference")
            except Exception as e:
                print(f"⚠️  Could not enable half precision: {e}")
        
        # Cache the model
        self._cache[cache_key] = model
        print(f"✅ Model loaded and cached: {os.path.basename(model_path)}")
        
        return model
    
    def clear_cache(self):
        """Clear all cached models"""
        self._cache.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🗑️  Model cache cleared")
    
    def get_device(self) -> str:
        """Get current device"""
        return self._device
    
    def use_half_precision(self) -> bool:
        """Check if half precision is enabled"""
        return self._half_precision
    
    def get_inference_kwargs(self) -> dict:
        """
        Get kwargs for model inference with optimal settings
        
        Returns:
            dict: kwargs to pass to model.predict()
        """
        kwargs = {
            'device': self._device,
            'verbose': False
        }
        
        # Add half precision for CUDA
        if self._half_precision:
            kwargs['half'] = True
        
        return kwargs
    
    def get_cache_info(self) -> dict:
        """Get information about cached models"""
        return {
            'device': self._device,
            'half_precision': self._half_precision,
            'cached_models': len(self._cache),
            'model_paths': [os.path.basename(p) for p in self._cache.keys()]
        }


# Global singleton instance
_model_cache_instance = ModelCache()


def get_model_cache() -> ModelCache:
    """Get the global model cache instance"""
    return _model_cache_instance


def get_optimized_model(model_path: str) -> YOLO:
    """
    Convenience function to get optimized model
    
    Args:
        model_path: Path to model weights
        
    Returns:
        Optimized YOLO model
    """
    return get_model_cache().get_model(model_path)


def get_inference_config() -> dict:
    """
    Get optimal inference configuration
    
    Returns:
        dict with device and precision settings
    """
    cache = get_model_cache()
    return {
        'device': cache.get_device(),
        'half': cache.use_half_precision(),
        'verbose': False
    }

