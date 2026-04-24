"""Visualization utilities for DOS-GCNN."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase import data as ase_data


def plot_dos(dos_pred, elems, atom_index, grid_size=400, component=1, 
             mode=0, output_dir=None):
    """Plot density of states.
    
    Args:
        dos_pred: Predicted DOS array
        elems: Element atomic numbers
        atom_index: Index of atom to plot
        grid_size: Number of grid points
        component: 1=s, 2=p, 3=d, 4=f
        mode: 0=individual component, 1=total atomic, 2=total crystal
        output_dir: Directory to save plots (optional)
    
    Returns:
        matplotlib figure
    """
    x_data = np.linspace(-10, 10, grid_size)
    y_data = dos_pred[atom_index]
    
    plt.figure(figsize=(6, 4))
    fig, ax1 = plt.subplots()
    ax1.plot(x_data, y_data, color='blue', label="Predicted by model")
    ax1.legend()
    
    plt.xlabel("Energy value, eV")
    plt.ylabel("Density of states")
    
    component_names = {1: "s-component", 2: "p-component", 
                       3: "d-component", 4: "f-component"}
    
    if mode == 0:
        elem_symbol = ase_data.chemical_symbols[elems[atom_index]]
        rr = component_names.get(component, "")
        plot_title = f"DoS for atom {atom_index} ({elem_symbol}): {rr}"
    elif mode == 1:
        elem_symbol = ase_data.chemical_symbols[elems[atom_index]]
        plot_title = f"Total DoS for atom {atom_index} ({elem_symbol})"
    else:
        plot_title = "Total density of states for crystal"
    
    plt.title(plot_title)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if mode == 0:
            comp_letter = ['s', 'p', 'd', 'f'][component - 1]
            filename = f"figure_{comp_letter}_atom_{atom_index}.png"
        elif mode == 1:
            filename = f"figure_atom_total_{atom_index}.png"
        else:
            filename = "figure_crystal_total.png"
        
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    
    return fig


def plot_all_dos(dos_s, dos_p, dos_d, dos_f, elements, output_dir=None):
    """Plot all DOS components for all atoms and total crystal DOS.
    
    Args:
        dos_s, dos_p, dos_d, dos_f: DOS arrays for s,p,d,f components
        elements: Element atomic numbers
        output_dir: Directory to save plots
    
    Returns:
        dict with all predictions
    """
    num_atoms = dos_s.shape[0]
    grid_size = dos_s.shape[1]
    
    total_crystal_dos = np.zeros((1, grid_size), dtype=np.float32)
    
    for atom_idx in range(num_atoms):
        # Plot individual components
        plot_dos(dos_s, elements, atom_idx, grid_size, 1, 0, output_dir)
        plot_dos(dos_p, elements, atom_idx, grid_size, 2, 0, output_dir)
        plot_dos(dos_d, elements, atom_idx, grid_size, 3, 0, output_dir)
        plot_dos(dos_f, elements, atom_idx, grid_size, 4, 0, output_dir)
        
        # Total atomic DOS
        total_atomic_dos = dos_s[atom_idx] + dos_p[atom_idx] + dos_d[atom_idx] + dos_f[atom_idx]
        total_atomic_array = np.zeros_like(dos_s)
        total_atomic_array[atom_idx] = total_atomic_dos
        plot_dos(total_atomic_array, elements, atom_idx, grid_size, 4, 1, output_dir)
        
        total_crystal_dos[0] += total_atomic_dos
    
    # Total crystal DOS
    plot_dos(total_crystal_dos, elements, 0, grid_size, 4, 2, output_dir)
    plt.close('all')
    
    return {
        'dos_s': dos_s,
        'dos_p': dos_p,
        'dos_d': dos_d,
        'dos_f': dos_f,
        'total_crystal': total_crystal_dos,
        'elements': elements
    }

