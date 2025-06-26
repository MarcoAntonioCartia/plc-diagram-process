import json

# Load the relative coordinate version
with open('D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction_relative.json', 'r') as f:
    relative_data = json.load(f)

print('Relative Coordinate System Structure:')
print('=' * 50)

# Show the first text region with relative coordinates
first_region = relative_data['text_regions'][0]
print('Example text region with relative coordinates:')
print(f'Text: "{first_region["text"]}"')
print(f'Coordinate system: {first_region.get("coordinate_system", "unknown")}')
print(f'Absolute bbox: {first_region.get("bbox_absolute", [])}')
print(f'Relative bbox: {first_region.get("bbox", [])}')
print(f'Symbol dimensions: {first_region.get("symbol_dimensions", {})}')
print(f'Relative position %: {first_region.get("relative_position_percent", {})}')

print('\nConversion metadata:')
metadata = relative_data.get('conversion_metadata', {})
for key, value in metadata.items():
    print(f'  {key}: {value}')

print('\nLayout data structure:')
with open('D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction_layout.json', 'r') as f:
    layout_data = json.load(f)

print(f'Layout coordinate system: {layout_data.get("coordinate_system")}')
print(f'Total symbols in layout: {layout_data["metadata"]["total_symbols"]}')
print(f'Total text regions in layout: {layout_data["metadata"]["total_text_regions"]}')

# Show first symbol with its text regions
first_symbol_id = list(layout_data['symbols'].keys())[0]
first_symbol = layout_data['symbols'][first_symbol_id]
print(f'\nFirst symbol: {first_symbol_id}')
print(f'  Symbol class: {first_symbol["symbol_info"]["class_name"]}')
print(f'  Text regions: {len(first_symbol["text_regions"])}')
if first_symbol["text_regions"]:
    first_text = first_symbol["text_regions"][0]
    print(f'  First text: "{first_text["text"]}"')
    print(f'  Relative bbox: {first_text["bbox_relative"]}')
    print(f'  Relative position %: {first_text["relative_position_percent"]}')
