import xml.etree.ElementTree as ET

file_path = '/home/dek/src/microtools/mindvision_python_app/src/mindvision_python_app/mainwindow.ui'

try:
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find camera_settings_tabs
    w = root.find(".//widget[@name='camera_settings_tabs']")
    if w is not None:
        print(f"Tag: {w.tag}")
        print(f"Attribs: {w.attrib}")
        print(f"Text: {w.text}")
    else:
        print("Widget not found")
        
except Exception as e:
    print(f"Error: {e}")
