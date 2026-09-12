# -*- coding: utf-8 -*-
"""Tests for the dependency-free namelist parser."""
from slvfe.namelist_parser import parse_namelist_text

SAMPLE_FEVARS = """
! sample parameters_fe
&fevars
  clcond = 'merge'          ! comment after value
  numprm = 11
  inptemp = 300.0
  uvread  = 'yes', slfslt = 'yes'
  norm_error = 1.0d-8
  force_calculation = .false.
  check_parameters_er = .true.
  large = 500000
  tiny = 1.0e-8
/

&other_group
  foo = 'bar'
/
"""

SAMPLE_ENE_PARAM = """
&ene_param
  ljformat = 1
  ljswitch = 0
  cmbrule = 0
  lwljcut = 10.0
  upljcut = 12.0
  hostspec = 5*0
  inptemp = 300, temp = 0.592
/
"""


def test_fevars():
    nml = parse_namelist_text(SAMPLE_FEVARS)
    assert set(nml.keys()) == {'fevars', 'other_group'}
    fe = nml['fevars']
    assert fe['clcond'] == 'merge'
    assert fe['numprm'] == 11
    assert fe['inptemp'] == 300.0
    assert fe['uvread'] == 'yes'
    assert fe['slfslt'] == 'yes'
    assert fe['norm_error'] == 1.0e-8
    assert fe['force_calculation'] is False
    assert fe['check_parameters_er'] is True
    assert fe['large'] == 500000
    assert fe['tiny'] == 1.0e-8
    assert nml['other_group']['foo'] == 'bar'
    print("test_fevars OK", fe)


def test_ene_param():
    nml = parse_namelist_text(SAMPLE_ENE_PARAM)
    ep = nml['ene_param']
    assert ep['ljformat'] == 1
    assert ep['lwljcut'] == 10.0
    assert ep['upljcut'] == 12.0
    assert ep['hostspec'] == [0, 0, 0, 0, 0]
    assert ep['inptemp'] == 300
    assert ep['temp'] == 0.592
    print("test_ene_param OK", ep)


if __name__ == '__main__':
    test_fevars()
    test_ene_param()
    print("ALL OK")
