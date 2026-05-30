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
        # Writes a minimal mdp file just to pass grompp validation
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
        """
        with open(path, 'w') as f:
            f.write(content.strip())

    def _link_ff_to_workdir(self):
        """Links the force field directory to the working directory so pdb2gmx finds it."""
        if self.matcher.ff_dir:
            ff_name = os.path.basename(self.matcher.ff_dir)
            target = os.path.join(self.gmx.work_dir, ff_name)
            if not os.path.exists(target):
                # Copy or symlink. Copy is safer cross-platform/docker
                shutil.copytree(self.matcher.ff_dir, target)
            return ff_name.replace(".ff", "")
        return None

    def run(self, input_pdb):
        self.report.append(f"Starting pipeline for: {input_pdb}")

        # 1. Parse
        try:
            struct = self.parser.parse_pdb(input_pdb)
            self.report.append("Successfully parsed structure.")
        except Exception as e:
            self.report.append(f"Error parsing structure: {e}")
            self.report.append("MD-ready: no")
            return self.report

        # 2. Match & Fix
        fixed_struct = self.matcher.fix_structure(struct, self.report)
        fixed_pdb_path = os.path.join(self.gmx.work_dir, "fixed.pdb")
        self.matcher.save_fixed_structure(fixed_struct, fixed_pdb_path)

        # 3. GROMACS Pipeline
        ff_name = self._link_ff_to_workdir()
        if not ff_name:
            self.report.append("Warning: Could not link local force field. Hoping it is installed system-wide.")
            ff_name = "charmm36" # Fallback guess

        gro_out = "complex.gro"
        top_out = "topol.top"

        success, out = self.gmx.pdb2gmx("fixed.pdb", gro_out, top_out, ff=ff_name)
        if not success:
            self.report.append(f"pdb2gmx failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report
        self.report.append("pdb2gmx passed")

        box_out = "complex_box.gro"
        success, out = self.gmx.editconf(gro_out, box_out)
        if not success:
            self.report.append(f"editconf failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report

        solv_out = "complex_solv.gro"
        success, out = self.gmx.solvate(box_out, top_out, solv_out)
        if not success:
            self.report.append(f"solvate failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report

        # Write dummy MDP for ions grompp
        mdp_path = os.path.join(self.gmx.work_dir, "ions.mdp")
        self.write_dummy_mdp(mdp_path)

        ions_tpr = "ions.tpr"
        success, out = self.gmx.grompp(solv_out, top_out, "ions.mdp", ions_tpr)
        if not success:
            self.report.append(f"grompp (ions) failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report

        ions_out = "complex_ions.gro"
        success, out = self.gmx.genion(ions_tpr, top_out, ions_out)
        if not success:
            self.report.append(f"genion failed:\n{out}")
            self.report.append("MD-ready: no")
            return self.report
        self.report.append("genion passed")

        # Final grompp to verify MD readiness
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
