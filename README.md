# UAAMD: Universal UAA-Aware MD Prep Pipeline

A universal UAA-aware MD prep pipeline that turns sequence/structure inputs into validated GROMACS-ready simulation files using auto-updated CHARMM36 force fields.

## Features
- Support for sequence+angles and PDB inputs.
- Auto-updates CHARMM36 force field from the MacKerell lab.
- Fixes atom and residue names to match force field expectations.
- End-to-end wrapper for GROMACS simulation preparation.
- Provides a comprehensive pipeline report.

## Installation
```bash
pip install .
```

## Usage
```bash
# Update Force Field
uaamd ff update charmm36

# Prepare MD system
uaamd prep --input-pdb file.pdb
```
