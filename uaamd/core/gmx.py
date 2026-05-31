import subprocess
import os

class GmxPipeline:
    """Wrapper for GROMACS command-line tools."""

    def __init__(self, work_dir):
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    def _run_cmd(self, cmd, input_str=None):
        """Runs a command and returns True if successful, False otherwise."""
        try:
            # We use 'gmx' as the command, assuming GROMACS is installed and sourced
            full_cmd = ["gmx"] + cmd
            result = subprocess.run(
                full_cmd,
                input=input_str,
                text=True,
                capture_output=True,
                cwd=self.work_dir
            )

            if result.returncode != 0:
                return False, result.stderr
            return True, result.stdout
        except FileNotFoundError:
            return False, "GROMACS ('gmx') is not installed or not in PATH."

    def pdb2gmx(self, input_pdb, output_gro, top_out, ff="charmm27", water="tip3p"):
        """Runs gmx pdb2gmx."""
        # Note: ff name must match the folder name in GROMACS share/gromacs/top or current dir
        cmd = [
            "pdb2gmx",
            "-f", input_pdb,
            "-o", output_gro,
            "-p", top_out,
            "-ff", ff,
            "-water", water,
            "-ignh" # Ignore hydrogens to let pdb2gmx rebuild them properly
        ]
        return self._run_cmd(cmd, input_str="0\n0\n")

    def editconf(self, input_gro, output_gro, box_type="cubic", distance=1.0):
        """Runs gmx editconf to define the box."""
        cmd = [
            "editconf",
            "-f", input_gro,
            "-o", output_gro,
            "-bt", box_type,
            "-d", str(distance)
        ]
        return self._run_cmd(cmd)

    def solvate(self, input_gro, top_file, output_gro):
        """Runs gmx solvate."""
        cmd = [
            "solvate",
            "-cp", input_gro,
            "-cs", "spc216.gro", # standard water box for tip3p
            "-o", output_gro,
            "-p", top_file
        ]
        return self._run_cmd(cmd)

    def genion(self, input_tpr, top_file, output_gro, pname="NA", nname="CL", conc=0.15):
        """Runs gmx genion to add ions."""
        cmd = [
            "genion",
            "-s", input_tpr,
            "-p", top_file,
            "-o", output_gro,
            "-pname", pname,
            "-nname", nname,
            "-neutral",
            "-conc", str(conc)
        ]
        # Need to provide input group (e.g. 13 for SOL, or generic "SOL")
        return self._run_cmd(cmd, input_str="SOL\n")

    def grompp(self, input_gro, top_file, mdp_file, output_tpr):
        """Runs gmx grompp."""
        cmd = [
            "grompp",
            "-f", mdp_file,
            "-c", input_gro,
            "-p", top_file,
            "-o", output_tpr,
            "-maxwarn", "2"
        ]
        return self._run_cmd(cmd)
