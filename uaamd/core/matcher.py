import os
import glob
from Bio.PDB import PDBIO

def find_latest_charmm36_dir(ff_base_dir):
    """Finds the best available force field directory."""
    # Preference order - newer is better
    preferences = [
        "charmm36m.ff",
        "charmm36-feb2021.ff", 
        "charmm36.ff",
        "charmm27.ff",
    ]
    for ff in preferences:
        candidate = os.path.join(ff_base_dir, ff)
        if os.path.isdir(candidate):
            return candidate
    # Fallback - any charmm ff
    ff_dirs = glob.glob(os.path.join(ff_base_dir, "charmm*.ff"))
    if ff_dirs:
        return sorted(ff_dirs, key=os.path.getmtime, reverse=True)[0]
    return None

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
                elif section in ('bonds', 'angles', 'dihedrals', 'impropers', 'cmap'):
                    in_atoms = False
                elif section != 'bondedtypes':
                    current_res = section
                    residues[current_res] = []
                    in_atoms = False
            elif in_atoms and current_res:
                parts = line.split()
                if parts:
                    residues[current_res].append(parts[0])

    return residues

class ForceFieldMatcher:
    """Matches residues/atoms to force field definitions and fixes names."""

    def __init__(self, ff_base_dir=os.path.expanduser("~/.uaamd/ff")):
        self.ff_dir = find_latest_charmm36_dir(ff_base_dir)
        self.ff_residues = {}

        self.mappings = {
            "F": "FE3",
            "HIS": "HSD",
            "SEP": {"P": "P", "O1P": "O1P", "O2P": "O2P", "O3P": "O3P"},
            "TPO": {"P": "P", "O1P": "O1P", "O2P": "O2P", "O3P": "O3P"},
            "PTR": {"P": "P", "O1P": "O1P", "O2P": "O2P", "O3P": "O3P"},
            "MLY": {"NZ": "NZ", "CH1": "CH1", "CH2": "CH2"},
            "MSE": {"SE": "SE", "CE": "CE"}
        }

        if self.ff_dir:
            self._load_rtp()

    def _load_rtp(self):
        rtp_files = glob.glob(os.path.join(self.ff_dir, "*.rtp"))
        for rtp in rtp_files:
            self.ff_residues.update(parse_rtp(rtp))

    def is_residue_supported(self, resname):
        return resname in self.ff_residues

    def fix_structure(self, structure, report_lines):
        """Fixes atom and residue names in the BioPython structure to match the force field."""
        if not self.ff_dir:
            report_lines.append("Warning: No force field directory found. Cannot fix names.")
            return structure

        fixed_count = 0
        missing_res_count = 0
        warned_residues = set()  # deduplicate warnings

        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()

                    if not self.is_residue_supported(resname):
                        missing_res_count += 1
                        if resname not in warned_residues:
                            report_lines.append(
                                f"Warning: Residue {resname} not found in force field database."
                            )
                            warned_residues.add(resname)
                        continue

                    ff_atoms = set(self.ff_residues[resname])

                    for atom in residue:
                        atom_name = atom.get_name()

                        if atom_name not in ff_atoms:
                            new_name = None
                            if resname in self.mappings and isinstance(self.mappings[resname], dict):
                                if atom_name in self.mappings[resname]:
                                    new_name = self.mappings[resname][atom_name]
                            elif atom_name in self.mappings and not isinstance(self.mappings[atom_name], dict):
                                new_name = self.mappings[atom_name]

                            if new_name and new_name in ff_atoms:
                                report_lines.append(
                                    f"Atom mismatch fixed in {resname}: {atom_name} -> {new_name}"
                                )
                                atom.set_name(new_name)
                                fixed_count += 1

        report_lines.append(f"Fixed {fixed_count} atom name mismatches.")
        if missing_res_count == 0:
            report_lines.append("All residues found in force field database.")

        return structure

    def save_fixed_structure(self, structure, out_filepath):
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_filepath)
        return out_filepath