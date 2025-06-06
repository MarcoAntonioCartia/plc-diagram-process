import svgwrite
from pathlib import Path

def make_grid_diagram(out_path, cols=3, rows=3, width=800, height=600):
    dwg = svgwrite.Drawing(str(out_path), profile='tiny')
    cell_w, cell_h = width/cols, height/rows
    for i in range(cols):
        for j in range(rows):
            x, y = i*cell_w, j*cell_h
            dwg.add(dwg.rect((x, y), (cell_w-10, cell_h-10), stroke='black', fill='none'))
    dwg.save()

if __name__ == "__main__":
    # Get configuration and build path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import get_config
    
    config = get_config()
    processed_data_path = Path(config.config['paths']['processed_data'])
    out = processed_data_path / "synthetic" / "diagram.svg"
    
    # Ensure the synthetic directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    make_grid_diagram(out)  # convert with rsvg-convert or cairosvg to PNG
