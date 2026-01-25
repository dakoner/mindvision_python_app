import sys
import os

# Add the directory containing the generated module to sys.path
# Using the path from main.py
release_dir = os.path.join(r"c:\users\davidek\microtools\mindvision_qobject", "release")
sys.path.insert(0, release_dir)

os.add_dll_directory(release_dir)
# Add Qt bin directory
os.add_dll_directory(r"C:\Qt\6.10.1\msvc2022_64\bin")
# Add MindVision SDK directory
os.add_dll_directory(r"C:\Program Files (x86)\MindVision\SDK\X64")

try:
    import _mindvision_qobject_py
    print("Module imported successfully.")
    
    print("\nAttributes of MindVisionCamera:")
    for attr in dir(_mindvision_qobject_py.MindVisionCamera):
        print(attr)
        
except ImportError as e:
    print(f"Failed to import: {e}")
