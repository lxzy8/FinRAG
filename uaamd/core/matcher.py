import os
import glob
from Bio.PDB import PDBIO

def find_latest_charmm36_dir(ff_base_dir):
    """Finds the most recently downloaded charmm36 force field directory."""
    ff_dirs = glob.glob(os.path.join(ff_base_dir, "charmm36*.ff"))
    if not ff_dirs:
        return None
    # Just take the first one found or sort by modification time
    return sorted(ff_dirs, key=os.path.getmtime, reverse=True)[0]

def parse_rtp(rtp_path):
    """Parses a GROMACS .rtp file to get residue definitions and atom names."""
    residues = {}
    current_res = None

    if not os.path.exists(rtp_path):
        return residues

    with open(rtp_path, 'r') as f:
        in_atoms = False
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1].strip()
                if section == 'atoms':
                    in_atoms = True
                elif section == 'bonds' or section == 'angles' or section == 'dihedrals' or section == 'impropers' or section == 'cmap':
                    in_atoms = False
                elif section != 'bondedtypes':
                    # New residue
                    current_res = section
                    residues[current_res] = []
                    in_atoms = False
            elif in_atoms and current_res:
                parts = line.split()
                if parts:
                    atom_name = parts[0]
                    residues[current_res].append(atom_name)

    return residues

class ForceFieldMatcher:
    """Matches residues/atoms to force field definitions and fixes names."""

    def __init__(self, ff_base_dir=os.path.expanduser("~/.uaamd/ff")):
        self.ff_dir = find_latest_charmm36_dir(ff_base_dir)
        self.ff_residues = {}

        if self.ff_dir:
            self._load_rtp()

    def _load_rtp(self):
        # CHARMM36 typically has aminoacids.rtp, nucleic.rtp, lipids.rtp, etc.
        rtp_files = glob.glob(os.path.join(self.ff_dir, "*.rtp"))
        for rtp in rtp_files:
            res_dict = parse_rtp(rtp)
            self.ff_residues.update(res_dict)

    def is_residue_supported(self, resname):
        return resname in self.ff_residues

    def fix_structure(self, structure, report_lines):
        """Fixes atom and residue names in the BioPython structure to match the force field."""
        if not self.ff_dir:
            report_lines.append("Warning: No force field directory found. Cannot fix names.")
            return structure

        fixed_count = 0
        missing_res_count = 0

        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()

                    if not self.is_residue_supported(resname):
                        missing_res_count += 1
                        report_lines.append(f"Warning: Residue {resname} not found in CHARMM36.")
                        continue

                    ff_atoms = set(self.ff_residues[resname])

                    # Fix atom names (simple heuristics)
                    for atom in residue:
                        # e.g., F -> FE3 mapping or standardizing hydrogen names
                        atom_name = atom.get_name()

                        if atom_name not in ff_atoms:
                            # Attempt simple mapping (this dictionary would grow based on UAA knowledge)
                            mappings = {
                                "F": "FE3",
                                "CD1": "CD1", # Example placeholders
                            }

                            if atom_name in mappings and mappings[atom_name] in ff_atoms:
                                report_lines.append(f"Atom mismatch fixed in {resname}: {atom_name} -> {mappings[atom_name]}")
                                atom.set_name(mappings[atom_name])
                                fixed_count += 1
                            else:
                                pass # Wait for grompp to complain if really missing

        report_lines.append(f"Fixed {fixed_count} atom name mismatches.")
        if missing_res_count == 0:
            report_lines.append("All residues found in CHARMM36.")

        return structure

    def save_fixed_structure(self, structure, out_filepath):
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_filepath)
        return out_filepath
