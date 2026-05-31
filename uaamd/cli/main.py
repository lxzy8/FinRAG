import click

@click.group()
def cli():
    """Universal UAA-aware MD Prep Pipeline."""
    pass

@cli.command()
def info():
    """Display information about UAAMD."""
    click.echo("UAAMD: Universal UAA-aware MD prep pipeline is running!")

@cli.group()
def ff():
    """Force field management."""
    pass

@ff.command()
@click.argument('ff_name', type=click.Choice(['charmm36', 'charmm27']))
def update(ff_name):
    """Update a specific force field."""
    from uaamd.ff.updater import update_ff
    update_ff(ff_name)

@cli.command()
@click.option('--input-pdb', help="Input PDB file.")
@click.option('--input-seq', help="Input Sequence+Angles file.")
@click.option('--input-smiles', help="Input SMILES string.")
@click.option('--work-dir', default="uaamd_out", help="Base output directory.")
def prep(input_pdb, input_seq, input_smiles, work_dir):
    """Run the MD preparation pipeline."""
    if not input_pdb and not input_seq and not input_smiles:
        click.echo("Error: Must provide either --input-pdb, --input-seq, or --input-smiles", err=True)
        return

    from uaamd.core.parser import StructureParser
    from uaamd.core.matcher import ForceFieldMatcher
    from uaamd.core.gmx import GmxPipeline
    from uaamd.core.pipeline import PipelineRunner
    import os
    import tempfile

    parser = StructureParser()

    # If a sequence is provided, parse it to a temporary PDB file first
    if input_seq and not input_pdb:
        click.echo(f"Building 3D structure from sequence: {input_seq}...")
        try:
            struct = parser.parse_sequence_angles(input_seq)

            # Save it temporarily to pass into the pipeline
            fd, temp_path = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            from Bio.PDB import PDBIO
            io = PDBIO()
            io.set_structure(struct)
            io.save(temp_path)

            input_pdb = temp_path
            click.echo(f"Generated temporary PDB at {input_pdb}")
        except Exception as e:
            click.echo(f"Error parsing sequence: {e}", err=True)
            return

    # If SMILES is provided, parse it to a temporary PDB file first
    if input_smiles and not input_pdb:
        click.echo(f"Building 3D structure from SMILES: {input_smiles}...")
        try:
            struct = parser.parse_smiles(input_smiles)

            # Save it temporarily to pass into the pipeline
            fd, temp_path = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            from Bio.PDB import PDBIO
            io = PDBIO()
            io.set_structure(struct)
            io.save(temp_path)

            input_pdb = temp_path
            click.echo(f"Generated temporary PDB at {input_pdb}")
        except Exception as e:
            click.echo(f"Error parsing SMILES: {e}", err=True)
            return

    click.echo(f"Initializing UAAMD pipeline for {input_pdb}...")

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    parser = StructureParser()
    matcher = ForceFieldMatcher()
    gmx = GmxPipeline(work_dir=work_dir)

    runner = PipelineRunner(parser, matcher, gmx)
    report = runner.run(input_pdb)

    click.echo("\n--- UAAMD Report ---")
    for line in report:
        click.echo(line)

    with open(os.path.join(work_dir, "uaamd_report.txt"), "w") as f:
        for line in report:
            f.write(line + "\n")

if __name__ == '__main__':
    cli()
