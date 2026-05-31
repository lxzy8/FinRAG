import os
import shutil


class PipelineRunner:
    """Orchestrates the entire preparation and validation pipeline."""

    def __init__(self, parser, matcher, gmx_wrapper):
        self.parser = parser
        self.matcher = matcher
        self.gmx = gmx_wrapper
        self.report = []

    def write_dummy_mdp(self, path):
        """Writes a minimal mdp file to pass grompp validation."""
        content = """
integrator               = md
nsteps                   = 1000
dt                       = 0.002
nstenergy                = 500
cutoff-scheme            = Verlet
coulombtype              = PME
rcoulomb                 = 1.2
rvdw                     = 1.2
pbc                      = xyz
constraints              = h-bonds
"""
        with open(path, 'w') as f:
            f.write(content.strip())

    def _link_ff_to_workdir(self):
        """Copies the force field directory into the working directory so pdb2gmx finds it."""
        if self.matcher.ff_dir:
            ff_name = os.path.basename(self.matcher.ff_dir)
            target = os.path.join(self.gmx.work_dir, ff_name)
            if not os.path.exists(target):
                shutil.copytree(self.matcher.ff_dir, target)
            return ff_name.replace(".ff", "")
        return None

    def _clean_pdb_text(self, input_pdb, output_pdb):
        """Text-level PDB cleaning — strips HOH and alternate conformations."""
        with open(input_pdb, 'r') as f:
            lines = f.readlines()

        with open(output_pdb, 'w') as f:
            for line in lines:
                # Skip crystal waters
                if line.startswith('HETATM') and 'HOH' in line:
                    continue
                # Handle alternate conformations
                if line.startswith(('ATOM', 'HETATM')) and len(line) > 16:
                    altloc = line[16]
                    if altloc not in (' ', 'A'):
                        continue
                    line = line[:16] + ' ' + line[17:]
                f.write(line)

        self.report.append("Stripped crystal waters and alternate conformations.")

    def run(self, input_pdb):
        self.report.append(f"Starting pipeline for: {input_pdb}")

        # Per-protein output folder
        protein_name = os.path.splitext(os.path.basename(input_pdb))[0]
        self.gmx.work_dir = os.path.join(self.gmx.work_dir, protein_name)
        os.makedirs(self.gmx.work_dir, exist_ok=True)

        # 1. Text-clean original PDB — preserve GROMACS-compatible format
        fixed_pdb_path = os.path.join(self.gmx.work_dir, "fixed.pdb")
        self._clean_pdb_text(input_pdb, fixed_pdb_path)

        # 2. Parse cleaned PDB with BioPython for analysis/reporting
        try:
            struct = self.parser.parse_pdb(fixed_pdb_path)
            self.report.append("Successfully parsed structure.")
        except Exception as e:
            self.report.append(f"Error parsing structure: {e}")
            self.report.append("MD-ready: no")
            return self.report

        # 3. Matcher — analysis and reporting only, does not overwrite fixed.pdb
        self.matcher.fix_structure(struct, self.report)

        # 4. Link force field
        ff_name = self._link_ff_to_workdir()
        if not ff_name:
            self.report.append("Warning: No force field found. Run 'uaamd ff update charmm27' first.")
            self.report.append("MD-ready: no")
            return self.report

        gro_out = "complex.gro"
        top_out = "topol.top"

        # 5. pdb2gmx
        success, out = self.gmx.pdb2gmx("fixed.pdb", gro_out, top_out, ff=ff_name)
        if not success:
            self.report.append(f"pdb2gmx failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report
        self.report.append("pdb2gmx passed")

        # 6. editconf — create simulation box
        box_out = "complex_box.gro"
        success, out = self.gmx.editconf(gro_out, box_out)
        if not success:
            self.report.append(f"editconf failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report

        # 7. solvate
        solv_out = "complex_solv.gro"
        success, out = self.gmx.solvate(box_out, top_out, solv_out)
        if not success:
            self.report.append(f"solvate failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report

        # 8. grompp for ions
        mdp_path = os.path.join(self.gmx.work_dir, "ions.mdp")
        self.write_dummy_mdp(mdp_path)
        ions_tpr = "ions.tpr"
        success, out = self.gmx.grompp(solv_out, top_out, "ions.mdp", ions_tpr)
        if not success:
            self.report.append(f"grompp (ions) failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report

        # 9. genion — neutralize system
        ions_out = "complex_ions.gro"
        success, out = self.gmx.genion(ions_tpr, top_out, ions_out)
        if not success:
            self.report.append(f"genion failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report
        self.report.append("genion passed")

        # 10. Final grompp — MD readiness check
        mdp_min_path = os.path.join(self.gmx.work_dir, "min.mdp")
        self.write_dummy_mdp(mdp_min_path)
        tpr_out = "min.tpr"
        success, out = self.gmx.grompp(ions_out, top_out, "min.mdp", tpr_out)
        if not success:
            self.report.append(f"grompp (final) failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report
        self.report.append("grompp passed")

        self.report.append("MD-ready: yes")
        return self.report