import os
import tempfile
from Bio.PDB import PDBParser

class StructureParser:
    """Parses initial structures from various inputs."""

    def __init__(self):
        pass

    def parse_pdb(self, filepath):
        """Parses a PDB file using BioPython."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDB file not found: {filepath}")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("system", filepath)
        return structure

    def parse_sequence_angles(self, seq_angles_file):
        """
        Parses custom sequence + angles file and builds a 3D structure.
        Format expectation:
        RESIDUE_NAME PHI PSI OMEGA
        """
        from Bio.PDB import PDBIO
        from Bio.PDB.Polypeptide import d1_to_index, d3_to_index
        from Bio.PDB.StructureBuilder import StructureBuilder
        import numpy as np

        if not os.path.exists(seq_angles_file):
            raise FileNotFoundError(f"Sequence angles file not found: {seq_angles_file}")

        import PeptideBuilder
        import Bio.PDB

        structure_seq = None

        with open(seq_angles_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue

                resname = parts[0]
                phi = float(parts[1]) if len(parts) > 1 else -120.0
                psi = float(parts[2]) if len(parts) > 2 else 140.0
                omega = float(parts[3]) if len(parts) > 3 else -180.0

                # Using 1-letter codes if standard, otherwise PeptideBuilder defaults
                # to creating a generic residue backbone where we might need to manually set the resname later
                # PeptideBuilder accepts 1 letter codes for standard amino acids
                one_letter = resname
                if len(resname) == 3:
                    one_letter = Bio.PDB.Polypeptide.protein_letters_3to1.get(resname, "G")

                # Ensure it's 1 letter for PeptideBuilder, fallback to Glycine backbone for non-standards
                # so the backbone geometry is constructed properly
                pb_res = one_letter if len(one_letter) == 1 else "G"

                geo = PeptideBuilder.Geometry.geometry(pb_res)
                geo.phi = phi
                geo.psi_im1 = psi
                geo.omega = omega

                if structure_seq is None:
                    # PeptideBuilder initialize_res takes a Geometry object since v1.1
                    structure_seq = PeptideBuilder.initialize_res(geo)
                else:
                    # PeptideBuilder add_residue takes the Structure and a Geometry object
                    PeptideBuilder.add_residue(structure_seq, geo)

        if structure_seq is None:
            raise ValueError("No residues found in sequence file.")

        # Write out to PDB format using BioPython
        fd, temp_path = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)

        io = Bio.PDB.PDBIO()
        io.set_structure(structure_seq)
        io.save(temp_path)

        # Now we parse it back in, and fix the residue names to match what was in the file
        # since PeptideBuilder might have used "GLY" backbones for the UAAs
        struct = self.parse_pdb(temp_path)

        residue_names = []
        with open(seq_angles_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    residue_names.append(parts[0])

        res_idx = 0
        for model in struct:
            for chain in model:
                for residue in chain:
                    if res_idx < len(residue_names):
                        residue.resname = residue_names[res_idx]
                        res_idx += 1

        return struct

    def parse_smiles(self, smiles_string):
        """
        Parses a SMILES string, generates 3D coordinates, and returns a BioPython structure.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import tempfile
        import os

        # 1. Parse SMILES to Mol
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES string: {smiles_string}")

        # 2. Add hydrogens
        mol = Chem.AddHs(mol)

        # 3. Generate 3D coordinates using ETKDG (standard RDKit embedding)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        # 4. Optimize geometry with MMFF
        AllChem.MMFFOptimizeMolecule(mol)

        # 5. Export to PDB format
        pdb_block = Chem.MolToPDBBlock(mol)

        # 6. Save to temp file and load via BioPython
        fd, temp_path = tempfile.mkstemp(suffix=".pdb")
        with os.fdopen(fd, 'w') as f:
            f.write(pdb_block)

        return self.parse_pdb(temp_path)
