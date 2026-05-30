import pytest
from click.testing import CliRunner
from uaamd.cli.main import cli
from uaamd.core.matcher import parse_rtp

def test_info_command():
    runner = CliRunner()
    result = runner.invoke(cli, ['info'])
    assert result.exit_code == 0
    assert "UAAMD: Universal UAA-aware MD prep pipeline is running!" in result.output

def test_parse_rtp_empty(tmp_path):
    empty_rtp = tmp_path / "test.rtp"
    empty_rtp.write_text("")
    res = parse_rtp(str(empty_rtp))
    assert res == {}

def test_parse_rtp_simple(tmp_path):
    rtp_content = """
[ bondedtypes ]
1 1 1 1 1 1

[ ALA ]
 [ atoms ]
  N    NH1   -0.47  0
  HN   H      0.31  0
  CA   CT1    0.07  1
 [ bonds ]
  N  HN

[ ARG ]
 [ atoms ]
  N    NH1   -0.47  0
"""
    test_rtp = tmp_path / "test.rtp"
    test_rtp.write_text(rtp_content)
    res = parse_rtp(str(test_rtp))
    assert "ALA" in res
    assert "ARG" in res
    assert res["ALA"] == ["N", "HN", "CA"]
    assert res["ARG"] == ["N"]

def test_parse_smiles():
    from uaamd.core.parser import StructureParser
    parser = StructureParser()
    struct = parser.parse_smiles("CCO")
    assert struct is not None
    atoms = list(struct.get_atoms())
    # CCO = 2 carbons, 1 oxygen, 6 hydrogens = 9 atoms
    assert len(atoms) == 9

def test_parse_sequence_angles(tmp_path):
    from uaamd.core.parser import StructureParser
    seq_file = tmp_path / "test.seq"
    seq_file.write_text("ALA -60 -40 180\nGLY -60 -40 180")
    parser = StructureParser()
    struct = parser.parse_sequence_angles(str(seq_file))
    assert struct is not None
    residues = list(struct.get_residues())
    assert len(residues) == 2
    assert residues[0].get_resname() == "ALA"
    assert residues[1].get_resname() == "GLY"
