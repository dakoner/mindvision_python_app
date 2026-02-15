import xml.etree.ElementTree as ET
import os

file_path = '/home/dek/src/microtools/mindvision_python_app/src/mindvision_python_app/mainwindow.ui'

try:
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    found = False
    # Find all widgets with sizePolicy attribute
    for widget in root.iter('widget'):
        if 'sizePolicy' in widget.attrib:
            print(f"Found sizePolicy in {widget.get('name')}: {widget.attrib['sizePolicy']}")
            del widget.attrib['sizePolicy']
            found = True
            
    if found:
        tree.write(file_path, encoding='UTF-8', xml_declaration=True)
        print("Removed sizePolicy attributes.")
    else:
        print("No sizePolicy attributes found.")
        
except Exception as e:
    print(f"Error: {e}")
