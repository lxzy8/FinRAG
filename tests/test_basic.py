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
