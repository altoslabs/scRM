#!/usr/bin/env python3
"""Unit tests for the cell-accounting arithmetic."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "cell_accounting", Path(__file__).with_name("cell_accounting.py")
)
cell_accounting = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cell_accounting)
audit = cell_accounting.audit

FLOOR = 500


def counts(**kw):
    base = {
        "cells_called": 100,
        "dropped_cells_shallow": 0,
        "dropped_cells_molecules": 0,
        "dropped_umi_short": 0,
        "dropped_umi_long": 0,
    }
    base.update(kw)
    return base


def test_a_corpus_that_lost_nothing_closes():
    assert audit(counts(), 100, FLOOR) == []


def test_differing_shallow_and_molecule_counters_close():
    """Differing shallow and molecule counters must close: 4,416 called, 2 shallow, 869 molecules."""
    c = counts(
        cells_called=4416,
        dropped_cells_shallow=2,
        dropped_cells_molecules=869,
        dropped_umi_short=651024,
        dropped_umi_long=59,
    )
    assert audit(c, 4414, FLOOR) == []


def test_a_large_shallow_drop_closes():
    """82 shallow cells carrying 33,508 molecules -- mean 408.6, inside the floor."""
    c = counts(
        cells_called=15618,
        dropped_cells_shallow=82,
        dropped_cells_molecules=33508,
        dropped_umi_short=1725884,
        dropped_umi_long=518,
    )
    assert audit(c, 15536, FLOOR) == []


def test_molecules_recorded_without_any_shallow_cell_is_a_problem():
    got = audit(counts(dropped_cells_molecules=869), 100, FLOOR)
    assert len(got) == 1 and "no cells were dropped as shallow" in got[0]


@pytest.mark.parametrize(
    "shallow,molecules",
    [
        (2, 1),
        (2, 1000),
        (2, 5000),
    ],
)
def test_the_shallow_molecule_pair_must_bracket(shallow, molecules):
    c = counts(dropped_cells_shallow=shallow, dropped_cells_molecules=molecules)
    got = audit(c, 100 - shallow, FLOOR)
    assert any("outside" in p for p in got), f"{shallow} cells / {molecules} molecules should not pass"


def test_the_bracket_admits_its_own_edges():
    """One molecule per cell is legal; one below the floor per cell is legal; the floor itself is not."""
    assert audit(counts(dropped_cells_shallow=2, dropped_cells_molecules=2), 98, FLOOR) == []
    assert audit(counts(dropped_cells_shallow=2, dropped_cells_molecules=999), 98, FLOOR) == []
    assert audit(counts(dropped_cells_shallow=2, dropped_cells_molecules=1000), 98, FLOOR) != []


def test_a_cell_vanishing_with_no_upstream_cause_is_a_problem():
    """A cell gone for no recorded reason must be reported as a problem."""
    got = audit(counts(), 97, FLOOR)
    assert len(got) == 1 and "unexplained" in got[0]


def test_a_cell_vanishing_is_accepted_when_the_bounds_could_have_emptied_it():
    assert audit(counts(dropped_umi_short=5000), 97, FLOOR) == []


def test_more_cells_than_were_called_is_a_problem():
    got = audit(counts(), 101, FLOOR)
    assert len(got) == 1 and "appeared from somewhere" in got[0]


def test_the_recorded_emptied_counter_is_cross_checked_when_present():
    """The recorded emptied counter must be cross-checked against the derivation when present."""
    c = counts(dropped_umi_short=5000, dropped_cells_emptied=3)
    assert audit(c, 97, FLOOR) == []
    wrong = counts(dropped_umi_short=5000, dropped_cells_emptied=1)
    got = audit(wrong, 97, FLOOR)
    assert len(got) == 1 and "dropped_cells_emptied=1" in got[0]


def test_absence_of_the_recorded_counter_costs_only_that_cross_check():
    """A corpus with no recorded emptied counter must still get every other assertion."""
    c = counts(dropped_umi_short=5000)
    assert "dropped_cells_emptied" not in c
    assert audit(c, 97, FLOOR) == []
    assert audit(counts(dropped_cells_molecules=7), 100, FLOOR) != []
