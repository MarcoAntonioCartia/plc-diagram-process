import json

# Load the text extraction data
with open('D:/MarMe/github/0.3/plc-data/processed/text_extraction/1150_text_extraction.json', 'r') as f:
    data = json.load(f)

print('Analyzing text-to-symbol relationship patterns...\n')

# Analyze a few examples to understand the coordinate relationship
for i, region in enumerate(data['text_regions'][:5]):
    text = region['text']
    text_bbox = region['bbox']
    symbol = region.get('associated_symbol', {})
    symbol_bbox_global = symbol.get('bbox_global', {})
    
    if isinstance(symbol_bbox_global, dict):
        symbol_bbox = [
            symbol_bbox_global.get('x1', 0), 
            symbol_bbox_global.get('y1', 0),
            symbol_bbox_global.get('x2', 0), 
            symbol_bbox_global.get('y2', 0)
        ]
    else:
        symbol_bbox = symbol_bbox_global
    
    # Calculate relative position of text within the symbol box
    if len(text_bbox) >= 4 and len(symbol_bbox) >= 4:
        # Text position relative to symbol's top-left corner
        rel_x1 = text_bbox[0] - symbol_bbox[0]
        rel_y1 = text_bbox[1] - symbol_bbox[1]
        rel_x2 = text_bbox[2] - symbol_bbox[0]
        rel_y2 = text_bbox[3] - symbol_bbox[1]
        
        # Symbol dimensions
        symbol_width = symbol_bbox[2] - symbol_bbox[0]
        symbol_height = symbol_bbox[3] - symbol_bbox[1]
        
        print(f'Text: "{text}"')
        print(f'  Symbol box: [{symbol_bbox[0]:.1f}, {symbol_bbox[1]:.1f}, {symbol_bbox[2]:.1f}, {symbol_bbox[3]:.1f}]')
        print(f'  Symbol size: {symbol_width:.1f} x {symbol_height:.1f}')
        print(f'  Text absolute: [{text_bbox[0]:.1f}, {text_bbox[1]:.1f}, {text_bbox[2]:.1f}, {text_bbox[3]:.1f}]')
        print(f'  Text relative: [{rel_x1:.1f}, {rel_y1:.1f}, {rel_x2:.1f}, {rel_y2:.1f}]')
        print(f'  Text as % of symbol: X={rel_x1/symbol_width*100:.1f}%-{rel_x2/symbol_width*100:.1f}%, Y={rel_y1/symbol_height*100:.1f}%-{rel_y2/symbol_height*100:.1f}%')
        print()

print("\nNow let's see if we can convert back from relative to absolute coordinates...")
print("=" * 70)

# Test the reverse conversion for the first text region
first_region = data['text_regions'][0]
text = first_region['text']
text_bbox = first_region['bbox']
symbol = first_region.get('associated_symbol', {})
symbol_bbox_global = symbol.get('bbox_global', {})

if isinstance(symbol_bbox_global, dict):
    symbol_bbox = [
        symbol_bbox_global.get('x1', 0), 
        symbol_bbox_global.get('y1', 0),
        symbol_bbox_global.get('x2', 0), 
        symbol_bbox_global.get('y2', 0)
    ]
else:
    symbol_bbox = symbol_bbox_global

# Calculate relative coordinates
rel_x1 = text_bbox[0] - symbol_bbox[0]
rel_y1 = text_bbox[1] - symbol_bbox[1]
rel_x2 = text_bbox[2] - symbol_bbox[0]
rel_y2 = text_bbox[3] - symbol_bbox[1]

# Convert back to absolute coordinates
reconstructed_x1 = symbol_bbox[0] + rel_x1
reconstructed_y1 = symbol_bbox[1] + rel_y1
reconstructed_x2 = symbol_bbox[0] + rel_x2
reconstructed_y2 = symbol_bbox[1] + rel_y2

print(f'Original text bbox: [{text_bbox[0]:.1f}, {text_bbox[1]:.1f}, {text_bbox[2]:.1f}, {text_bbox[3]:.1f}]')
print(f'Reconstructed bbox: [{reconstructed_x1:.1f}, {reconstructed_y1:.1f}, {reconstructed_x2:.1f}, {reconstructed_y2:.1f}]')
print(f'Difference: [{abs(text_bbox[0] - reconstructed_x1):.1f}, {abs(text_bbox[1] - reconstructed_y1):.1f}, {abs(text_bbox[2] - reconstructed_x2):.1f}, {abs(text_bbox[3] - reconstructed_y2):.1f}]')

if (abs(text_bbox[0] - reconstructed_x1) < 0.1 and 
    abs(text_bbox[1] - reconstructed_y1) < 0.1 and 
    abs(text_bbox[2] - reconstructed_x2) < 0.1 and 
    abs(text_bbox[3] - reconstructed_y2) < 0.1):
    print("✓ Perfect reconstruction! Relative coordinate approach is viable.")
else:
    print("✗ Reconstruction failed. Need to investigate further.")
