### Project structure

```
bokhan/
├── main.py                      # CLI entry point
├── dos_gcnn/                    # Main package
│   ├── __init__.py              # Exports predict_dos, Config
│   ├── config.py                # Configuration with relative paths
│   ├── inference.py             # Main predict_dos() function
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # StructureDataset
│   │   └── processing.py        # process_data_spdf, get_dataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kan.py               # KANLinear
│   │   ├── gc_block.py          # GC_block
│   │   └── dos_predict.py       # DOSpredict model
│   └── utils/
│       ├── __init__.py
│       ├── data.py              # Data helper functions
│       └── visualization.py     # DOS visualization
└── bulk_new/                    # Data, models, config (unchanged)
```

### Usage

#### Simple function call

```python
from dos_gcnn import predict_dos

result = predict_dos("path/to/structure.cif")

# The result contains:
print(result.dos_s.shape)          # (n_atoms, 400) — s orbitals
print(result.dos_p.shape)          # (n_atoms, 400) — p orbitals
print(result.dos_d.shape)          # (n_atoms, 400) — d orbitals
print(result.dos_f.shape)          # (n_atoms, 400) — f orbitals
print(result.total_crystal_dos)    # Full crystal DOS
print(result.energy_grid)          # Energy grid (-10 to 10 eV)
print(result.ad)                   # Applicability-domain payload or None
```

#### With plots

```python
result = predict_dos("structure.cif", plot_results=True, output_dir="./plots")
```

#### Via the command line

```bash
python main.py bulk_new/tmp/test.cif --plot --output ./results
```

#### With a custom configuration

```python
from dos_gcnn import predict_dos, Config

config = Config(
    model_name="model_bulk_lorentz_28_spdf_terms_new.pth",
    graph_conv_type="Transformer",
)
result = predict_dos("structure.cif", config=config)
```
