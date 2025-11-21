"""
Cross-platform path utilities for portable file operations
"""
import os
import sys
from pathlib import Path
from typing import Union

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def get_project_root() -> Path:
    """
    Get the project root directory
    
    Returns:
        Path: Project root directory
    """
    return PROJECT_ROOT


def get_absolute_path(relative_path: Union[str, Path]) -> str:
    """
    Convert relative path to absolute path from project root
    
    Args:
        relative_path: Relative path from project root
        
    Returns:
        str: Absolute path
    """
    return str(PROJECT_ROOT / relative_path)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory: Directory path
        
    Returns:
        Path: Directory path object
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_data_path(filename: str = "") -> str:
    """
    Get path to file in data directory
    
    Args:
        filename: Filename (optional)
        
    Returns:
        str: Full path to data file/directory
    """
    if filename:
        return get_absolute_path(f"data/{filename}")
    return get_absolute_path("data")


def get_model_path(model_name: str = "best.pt") -> str:
    """
    Get path to model file
    
    Args:
        model_name: Model filename
        
    Returns:
        str: Full path to model
    """
    # Check default training location first
    default_path = get_absolute_path(f"runs/train/exp/weights/{model_name}")
    if os.path.exists(default_path):
        return default_path
    
    # Check root directory
    root_path = get_absolute_path(model_name)
    if os.path.exists(root_path):
        return root_path
    
    # Return default path even if doesn't exist (will cause error later with clear message)
    return default_path


def get_output_path(filename: str = "") -> str:
    """
    Get path to file in output directory
    
    Args:
        filename: Filename (optional)
        
    Returns:
        str: Full path to output file/directory
    """
    output_dir = ensure_dir(get_absolute_path("output"))
    if filename:
        return str(output_dir / filename)
    return str(output_dir)


def get_temp_path(filename: str) -> str:
    """
    Get path to temporary file
    
    Args:
        filename: Temporary filename
        
    Returns:
        str: Full path to temp file
    """
    temp_dir = ensure_dir(get_absolute_path("temp"))
    return str(temp_dir / filename)


def get_input_path(filename: str = "") -> str:
    """
    Get path to file in input directory
    
    Args:
        filename: Filename (optional)
        
    Returns:
        str: Full path to input file/directory
    """
    input_dir = ensure_dir(get_absolute_path("input"))
    if filename:
        return str(input_dir / filename)
    return str(input_dir)


def safe_file_write(filepath: Union[str, Path], content: bytes, mode: str = 'wb'):
    """
    Safely write file, ensuring directory exists
    
    Args:
        filepath: Target file path
        content: Content to write
        mode: File open mode
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, mode) as f:
        f.write(content)


def safe_file_read(filepath: Union[str, Path], mode: str = 'rb'):
    """
    Safely read file
    
    Args:
        filepath: Source file path
        mode: File open mode
        
    Returns:
        File content
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, mode) as f:
        return f.read()


def cleanup_temp_files():
    """Remove all temporary files"""
    temp_dir = Path(get_absolute_path("temp"))
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            try:
                if file.is_file():
                    file.unlink()
            except Exception as e:
                print(f"Could not delete {file}: {e}")


# Commonly used paths
DATA_DIR = get_data_path()
OUTPUT_DIR = get_output_path()
INPUT_DIR = get_input_path()

