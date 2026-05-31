import os
import shutil
import logging

SYSTEM_FF_PATHS = [
    "/usr/share/gromacs/top",
    "/usr/local/share/gromacs/top",
    "/opt/gromacs/share/gromacs/top",
]

FF_PREFERENCE = [
    "charmm36m.ff",
    "charmm36-feb2021.ff",
    "charmm36.ff",
    "charmm27.ff",
]

def find_system_ff(ff_name):
    """Find a specific force field in system GROMACS paths."""
    for base in SYSTEM_FF_PATHS:
        candidate = os.path.join(base, ff_name)
        if os.path.isdir(candidate):
            return candidate
    return None

def update_ff(ff_name="charmm27", ff_base_dir=os.path.expanduser("~/.uaamd/ff")):
    """
    Copy force field from system GROMACS installation to ~/.uaamd/ff/
    Prefers newer CHARMM36 variants, falls back to CHARMM27.
    """
    os.makedirs(ff_base_dir, exist_ok=True)

    # If user asked for charmm27 specifically
    if ff_name == "charmm27":
        candidates = ["charmm27.ff"]
    else:
        # Try charmm36 variants first, fallback to charmm27
        candidates = FF_PREFERENCE

    for ff_folder in candidates:
        system_path = find_system_ff(ff_folder)
        if system_path:
            dest = os.path.join(ff_base_dir, ff_folder)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(system_path, dest)
            print(f"Force field '{ff_folder}' copied from {system_path}")
            print(f"Installed to: {dest}")
            print("Force field update completed successfully.")
            return dest

    print("ERROR: No compatible CHARMM force field found in system GROMACS installation.")
    print("Make sure GROMACS is installed: sudo apt-get install gromacs")
    return None

def get_installed_ff(ff_base_dir=os.path.expanduser("~/.uaamd/ff")):
    """Returns the best available installed FF path."""
    for ff_folder in FF_PREFERENCE:
        candidate = os.path.join(ff_base_dir, ff_folder)
        if os.path.isdir(candidate):
            return candidate
    return None