"""
Coordination polyhedra analysis using pymatgen.

Computes coordination environments for transition metals and exports
polyhedra geometry (vertices, faces, colors) for 3Dmol.js visualization.

Reference:
    Coordination polyhedra were constructed using CrystalNN algorithm
    from pymatgen, based on Voronoi decomposition and solid angle weighting.
"""

import json
import numpy as np
from typing import Optional
from pathlib import Path

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.local_env import CrystalNN
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("Warning: pymatgen not installed. Polyhedra analysis disabled.")


# Transition metals that typically form coordination polyhedra
TRANSITION_METALS = {
    # 3d
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    # 4d
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
    # 5d
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
}

# Ligand atoms (typically O, F, Cl, S in oxides/halides)
LIGAND_ATOMS = {'O', 'F', 'Cl', 'S', 'N'}

# Color palette for different metal centers (hex colors)
METAL_COLORS = {
    'Co': '#3498db',  # Blue
    'Ir': '#9b59b6',  # Purple
    'Fe': '#e74c3c',  # Red
    'Mn': '#f39c12',  # Orange
    'Ti': '#1abc9c',  # Teal
    'Nb': '#2ecc71',  # Green
    'Mo': '#e91e63',  # Pink
    'Cr': '#00bcd4',  # Cyan
    'Ni': '#8bc34a',  # Light green
    'Cu': '#ff5722',  # Deep orange
    'Ru': '#673ab7',  # Deep purple
    'Rh': '#ffc107',  # Amber
    'Ta': '#795548',  # Brown
    'W':  '#607d8b',  # Blue grey
    'Os': '#4caf50',  # Green
    'Pt': '#9e9e9e',  # Grey
    'default': '#607d8b',
}


def hex_to_rgb(hex_color: str) -> dict:
    """Convert hex color to RGB dict (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return {'r': r, 'g': g, 'b': b}


def compute_convex_hull_faces(vertices: list[dict]) -> list[list[int]]:
    """
    Compute convex hull triangular faces from vertices.
    
    Uses scipy's ConvexHull for robust computation.
    Returns list of face indices (triangles).
    """
    from scipy.spatial import ConvexHull
    
    if len(vertices) < 4:
        return []
    
    points = np.array([[v['x'], v['y'], v['z']] for v in vertices])
    
    try:
        hull = ConvexHull(points)
        # ConvexHull.simplices gives triangular faces
        return hull.simplices.tolist()
    except Exception as e:
        print(f"ConvexHull failed: {e}")
        return []


def analyze_coordination_polyhedra(
    cif_path: str,
    metal_elements: Optional[set] = None,
    ligand_elements: Optional[set] = None,
    min_coordination: int = 4,
    max_coordination: int = 12,
) -> dict:
    """
    Analyze coordination polyhedra in crystal structure.
    
    Uses pymatgen's CrystalNN for robust nearest-neighbor analysis
    with Voronoi decomposition and solid angle weighting.
    
    Args:
        cif_path: Path to CIF file
        metal_elements: Set of metal elements to analyze (default: TRANSITION_METALS)
        ligand_elements: Set of ligand elements (default: LIGAND_ATOMS)
        min_coordination: Minimum coordination number for valid polyhedron
        max_coordination: Maximum coordination number
        
    Returns:
        dict with 'polyhedra' list and 'atoms' list for visualization
    """
    if not HAS_PYMATGEN:
        return {'polyhedra': [], 'atoms': [], 'error': 'pymatgen not installed'}
    
    metal_elements = metal_elements or TRANSITION_METALS
    ligand_elements = ligand_elements or LIGAND_ATOMS
    
    # Load structure
    structure = Structure.from_file(cif_path)
    
    # Add oxidation states for better CrystalNN results
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer
        bva = BVAnalyzer()
        structure = bva.get_oxi_state_decorated_structure(structure)
        print("Oxidation states assigned successfully")
    except Exception as e:
        # Fallback: try simple oxidation state guess
        try:
            structure.add_oxidation_state_by_guess()
            print("Oxidation states guessed")
        except Exception:
            print(f"Warning: Could not assign oxidation states: {e}")
    
    # Initialize CrystalNN analyzer
    cnn = CrystalNN(
        weighted_cn=True,
        cation_anion=True,  # Use cation-anion mode for oxides
        distance_cutoffs=(0.5, 6.0),
        x_diff_weight=3.0,
    )
    
    polyhedra = []
    atoms_data = []
    
    # Get all atom positions for visualization
    for i, site in enumerate(structure):
        elem = str(site.specie.element) if hasattr(site.specie, 'element') else str(site.specie)
        # Clean element name (remove oxidation state)
        elem = ''.join(c for c in elem if c.isalpha())
        
        cart_coords = site.coords.tolist()
        atoms_data.append({
            'index': i,
            'element': elem,
            'x': cart_coords[0],
            'y': cart_coords[1],
            'z': cart_coords[2],
            'is_metal': elem in metal_elements,
            'is_ligand': elem in ligand_elements,
        })
    
    # Analyze coordination for each metal site
    for i, site in enumerate(structure):
        elem = str(site.specie.element) if hasattr(site.specie, 'element') else str(site.specie)
        elem = ''.join(c for c in elem if c.isalpha())
        
        if elem not in metal_elements:
            continue
        
        try:
            # Get coordination info from CrystalNN
            nn_info = cnn.get_nn_info(structure, i)
            
            # Filter for ligand atoms only
            ligand_neighbors = [
                nn for nn in nn_info
                if ''.join(c for c in str(nn['site'].specie) if c.isalpha()) in ligand_elements
            ]
            
            coord_num = len(ligand_neighbors)
            
            if coord_num < min_coordination or coord_num > max_coordination:
                continue
            
            # Get vertices - store both index and relative offset from center
            # JavaScript will use atom positions from 3Dmol.js + offset for periodic images
            vertices = []
            vertex_indices = []
            center_cart = site.coords
            
            for nn in ligand_neighbors:
                neighbor_site = nn['site']
                image = np.array(nn.get('image', [0, 0, 0]))
                
                # Find the index of this neighbor in the structure
                neighbor_index = nn.get('site_index', None)
                if neighbor_index is None:
                    # Try to find by matching coords
                    for idx, s in enumerate(structure):
                        if np.allclose(s.frac_coords, neighbor_site.frac_coords, atol=0.01):
                            neighbor_index = idx
                            break
                
                # Calculate Cartesian coords with periodic image
                neighbor_frac = neighbor_site.frac_coords + image
                cart = structure.lattice.get_cartesian_coords(neighbor_frac)
                
                # Calculate offset from center (for periodic boundary handling)
                offset = cart - center_cart
                
                ligand_elem = ''.join(c for c in str(neighbor_site.specie) if c.isalpha())
                vertices.append({
                    'atom_index': neighbor_index,
                    'element': ligand_elem,
                    'image': [int(x) for x in image],
                    # Store offset from center atom (in Angstroms)
                    'offset_x': float(offset[0]),
                    'offset_y': float(offset[1]),
                    'offset_z': float(offset[2]),
                })
                vertex_indices.append(neighbor_index)
            
            # Compute convex hull faces
            faces = compute_convex_hull_faces(vertices)
            
            if not faces:
                continue
            
            # Get color for this metal
            color_hex = METAL_COLORS.get(elem, METAL_COLORS['default'])
            color_rgb = hex_to_rgb(color_hex)
            
            center_coords = site.coords.tolist()
            
            polyhedra.append({
                'center_index': i,
                'center_element': elem,
                'center': {
                    'x': center_coords[0],
                    'y': center_coords[1],
                    'z': center_coords[2],
                },
                'coordination': coord_num,
                'vertices': vertices,
                'faces': faces,  # List of [i, j, k] vertex indices
                'color': color_rgb,
                'color_hex': color_hex,
            })
            
            print(f"Found {elem} polyhedron at site {i}: CN={coord_num}")
            
        except Exception as e:
            print(f"Error analyzing site {i} ({elem}): {e}")
            continue
    
    # Get unit cell parameters for visualization
    lattice = structure.lattice
    cell_params = {
        'a': float(lattice.a),
        'b': float(lattice.b),
        'c': float(lattice.c),
        'alpha': float(lattice.alpha),
        'beta': float(lattice.beta),
        'gamma': float(lattice.gamma),
        'matrix': [[float(x) for x in row] for row in lattice.matrix],
    }
    
    # Ensure all numpy types are converted to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        return obj
    
    result = {
        'polyhedra': polyhedra,
        'atoms': atoms_data,
        'cell': cell_params,
        'num_polyhedra': len(polyhedra),
    }
    
    return convert_numpy(result)


def export_polyhedra_json(cif_path: str, output_path: Optional[str] = None) -> str:
    """
    Analyze CIF and export polyhedra to JSON.
    
    Args:
        cif_path: Path to CIF file
        output_path: Optional output JSON path
        
    Returns:
        JSON string with polyhedra data
    """
    result = analyze_coordination_polyhedra(cif_path)
    
    json_str = json.dumps(result, indent=2)
    
    if output_path:
        Path(output_path).write_text(json_str)
        print(f"Exported polyhedra to {output_path}")
    
    return json_str


# CLI usage
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python polyhedra.py <cif_file> [output.json]")
        sys.exit(1)
    
    cif_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    json_data = export_polyhedra_json(cif_path, output_path)
    if not output_path:
        print(json_data)

