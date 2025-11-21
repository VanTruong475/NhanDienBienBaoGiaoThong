"""
Cross-platform font utilities
"""
import os
import sys
from PIL import ImageFont
from typing import Optional

def get_system_font_paths():
    """
    Get system font directories based on OS
    
    Returns:
        list: List of font directory paths
    """
    font_paths = []
    
    if sys.platform == 'win32':
        # Windows
        font_paths.extend([
            r'C:\Windows\Fonts',
            os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        ])
    elif sys.platform == 'darwin':
        # macOS
        font_paths.extend([
            '/Library/Fonts',
            '/System/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ])
    else:
        # Linux and others
        font_paths.extend([
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts')
        ])
    
    return font_paths


def find_font_file(font_names=None):
    """
    Find a font file from a list of preferred fonts
    
    Args:
        font_names: List of font names to search for (without extension)
        
    Returns:
        str or None: Path to font file if found
    """
    if font_names is None:
        # Default font preferences
        if sys.platform == 'win32':
            font_names = ['arial', 'times', 'calibri', 'segoeui']
        elif sys.platform == 'darwin':
            font_names = ['Arial', 'Helvetica', 'Times New Roman', 'Courier']
        else:
            font_names = ['DejaVuSans', 'FreeSans', 'LiberationSans', 'Arial']
    
    font_paths = get_system_font_paths()
    font_extensions = ['.ttf', '.otf', '.TTF', '.OTF']
    
    for font_path in font_paths:
        if not os.path.exists(font_path):
            continue
            
        for font_name in font_names:
            for ext in font_extensions:
                # Try exact match
                full_path = os.path.join(font_path, f"{font_name}{ext}")
                if os.path.exists(full_path):
                    return full_path
                
                # Try case-insensitive search
                if os.path.isdir(font_path):
                    try:
                        for file in os.listdir(font_path):
                            if file.lower() == f"{font_name.lower()}{ext.lower()}":
                                return os.path.join(font_path, file)
                    except (PermissionError, OSError):
                        continue
    
    return None


def get_font(font_size=24, font_name=None):
    """
    Get a PIL ImageFont with cross-platform support
    
    Args:
        font_size: Font size in pixels
        font_name: Specific font name to use (optional)
        
    Returns:
        ImageFont: PIL font object
    """
    # Try to find custom font
    if font_name:
        font_path = find_font_file([font_name])
        if font_path:
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                print(f"⚠️  Could not load font {font_name}: {e}")
    
    # Try to find default system font
    font_path = find_font_file()
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"⚠️  Could not load system font: {e}")
    
    # Fallback to PIL default font
    print(f"ℹ️  Using PIL default font (size may not match {font_size}px)")
    return ImageFont.load_default()


# Cache fonts to avoid repeated loading
_font_cache = {}

def get_cached_font(font_size=24, font_name=None):
    """
    Get cached font or load new one
    
    Args:
        font_size: Font size in pixels
        font_name: Specific font name to use (optional)
        
    Returns:
        ImageFont: PIL font object
    """
    cache_key = (font_size, font_name)
    
    if cache_key not in _font_cache:
        _font_cache[cache_key] = get_font(font_size, font_name)
    
    return _font_cache[cache_key]

