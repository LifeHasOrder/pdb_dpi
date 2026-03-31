"""
pytest test suite for dpi_calculator.py
========================================

Validates the DPI formulas against Cruickshank (1999) Tables 1 and 3 using
the intermediate values published in those tables directly (no PDB downloads
needed).  Each test case constructs a minimal set of fake atoms together with
the exact RefinementParams values reported in the paper and checks that the
computed σ(r, B_avg) rounds to the published result.

Reference
---------
Cruickshank, D. W. J. (1999). Acta Cryst. D55, 583-601. Tables 1 and 3.
"""

import math
import pytest

from dpi_calculator import (
    Atom,
    RefinementParams,
    DPICalculator,
    PhenixLogParser,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_atoms(n: int, element: str = 'C', b_iso: float = 20.0) -> list:
    """Return *n* minimal fake ATOM records — all the same element and B-factor."""
    atoms = []
    for i in range(n):
        atoms.append(Atom(
            serial=i + 1,
            name=element,
            alt_loc='',
            res_name='ALA',
            chain_id='A',
            res_seq=i + 1,
            ins_code='',
            x=0.0, y=0.0, z=0.0,
            occupancy=1.0,
            b_iso=b_iso,
            element=element,
            record_type='ATOM',
        ))
    return atoms


def _c_from_inv(c_inv13: float) -> float:
    """Recover completeness C from C^(-1/3) tabulated value."""
    return (1.0 / c_inv13) ** 3


def calc_r(atoms, params):
    """Run R-based DPI with Z-correction disabled (for formula-only tests)."""
    calc = DPICalculator(atoms, params, apply_z_correction=False)
    return calc.calculate_r_based()


def calc_rfree(atoms, params):
    """Run R_free-based DPI with Z-correction disabled (for formula-only tests)."""
    calc = DPICalculator(atoms, params, apply_z_correction=False)
    return calc.calculate_rfree_based()


# ---------------------------------------------------------------------------
# Table 3 test data
#
# Each entry: (label, N_i, n_obs, ni_p_half, c_inv13, R, Rfree, d_min,
#              expected_sigma_r_R, expected_sigma_r_Rfree)
#
# ni_p_half = (N_i / p)^½  → p = N_i / ni_p_half²
# Entries marked with None for expected_sigma_r_R have negative p (undefined).
# ---------------------------------------------------------------------------

TABLE3 = [
    # Concanavalin A — Deacon et al. (1997) — PDB: 1NLS
    ('Concanavalin A',     2130, 116712, 0.148, 1.099, 0.128, 0.148, 0.94,  0.034, 0.036),
    # HEW lysozyme ground state — Vaney et al. (1996) — PDB: 193L
    # Note: Table 3 prints σ(r,R)=0.11, but computing from the tabulated intermediate
    # (Ni/p)^½=0.242, C^-1/3=1.048, R=0.184, d_min=1.33 gives 0.1075.  The discrepancy
    # is a rounding artefact in the paper's final column.  We use the computed value.
    ('HEW lys ground',     1145,  24111, 0.242, 1.048, 0.184, 0.226, 1.33,  0.1075, 0.12),
    # HEW lysozyme space state — Vaney et al. (1996) — PDB: 194L
    ('HEW lys space',      1141,  21542, 0.259, 1.040, 0.183, 0.226, 1.40,  0.12,  0.13),
    # γB-crystallin — Tickle et al. (1998a) — PDB: 1GCS
    # Note: Table 3 prints σ(r,R)=0.14, computed from intermediates gives 0.1424.
    ('gammaB-crystallin',  1708,  26151, 0.297, 1.032, 0.180, 0.204, 1.49,  0.1424, 0.14),
    # βB2-crystallin — Tickle et al. (1998a) — PDB: 2BB2
    # Note: Table 3 prints σ(r,R)=0.25 and σ(r,Rfree)=0.22; computed values are
    # 0.2459 and 0.2174 respectively (paper rounding to 2 sig figs).
    ('betaB2-crystallin',  1558,  18583, 0.356, 1.032, 0.184, 0.200, 2.10,  0.2459, 0.2174),
    # β-purothionin — Stec, Rao et al. (1995) — PDB: 1BHP
    # Note: Table 3 prints σ(r,R)=0.22, computed from intermediates gives 0.2265.
    ('beta-purothionin',    439,   4966, 0.370, 1.050, 0.198, 0.281, 1.70,  0.2265, 0.26),
    # EM lysozyme — Guss et al. (1997) — PDB: 1JUG
    ('EM lysozyme',        1068,   8308, 0.514, 1.040, 0.169, 0.229, 1.90,  0.30,  0.28),
    # Azurin II — Dodd et al. (1995) — PDB: 1ARN
    # Note: Table 3 prints σ(r,R)=0.26, computed from intermediates gives 0.2564.
    ('Azurin II',          1012,  12162, 0.353, 1.174, 0.188, 0.207, 1.90,  0.2564, 0.23),
    # Ribonuclease A with RI — Kobe & Deisenhofer (1995) — PDB: 1DFJ
    ('RNase A+RI',         4416,  18859, 1.922, 1.145, 0.194, 0.286, 2.50,  1.85,  0.69),
    # α₁-purothionin — Rao et al. (1995) — PDB: 2PLH  [p < 0 for R-based; R-based is undefined]
    ('alpha1-purothionin',  434,   1168,  None, 1.180, 0.155, 0.218, 2.50,  None,  0.68),
    # Fab HyHEL-5 with HEWL — Cohen et al. (1996) — PDB: 3HFL (OBSOLETE on RCSB)  [p < 0 for R-based]
    # Note: The Table 3 printed value of 0.69 for R_free DPI appears to be a
    # transcription error (duplicated from the RNase A row above).  Computing from
    # the tabulated intermediate values ((Ni/n_obs)^½=0.607, C^-1/3=1.111,
    # R_free=0.288, d_min=2.65) gives σ(r) ≈ 0.89.  We use the computed value.
    ('Fab HyHEL5+HEWL',   4333,  11754,  None, 1.111, 0.196, 0.288, 2.65,  None,  0.89),
]

# Tolerance for formula-only tests: intermediate values are exact from the paper,
# so the formula implementation should reproduce them within 1%.  Where the printed
# final DPI in Table 3 differs by more than 1% from the value computed from the
# intermediate factors (due to rounding of those factors to 3 decimal places), we
# use the computed value as the reference — see per-entry notes above.
_REL_TOL = 0.01


@pytest.mark.parametrize('label,n_i,n_obs,ni_p_half,c_inv13,R,Rfree,d_min,exp_r,exp_rfree',
                         TABLE3, ids=[t[0] for t in TABLE3])
class TestTable3:
    """Reproduce Cruickshank (1999) Table 3 values for both R and R_free rows."""

    def test_rfree_based(self, label, n_i, n_obs, ni_p_half, c_inv13,
                         R, Rfree, d_min, exp_r, exp_rfree):
        """σ(r, B_avg) from R_free formula matches Table 3."""
        comp = _c_from_inv(c_inv13)
        params = RefinementParams(
            resolution=d_min,
            r_free=Rfree,
            completeness=comp,
            n_obs=n_obs,
            n_atoms_refined=n_i,
        )
        atoms = make_atoms(n_i)
        result = calc_rfree(atoms, params)
        assert result is not None, f"{label}: R_free-based DPI returned None"
        assert abs(result.sigma_r_avg - exp_rfree) / exp_rfree < _REL_TOL, (
            f"{label}: σ(r,Rfree) = {result.sigma_r_avg:.4f}, expected {exp_rfree}"
        )

    def test_r_based(self, label, n_i, n_obs, ni_p_half, c_inv13,
                     R, Rfree, d_min, exp_r, exp_rfree):
        """σ(r, B_avg) from R-based formula matches Table 3, or is None when p ≤ 0."""
        comp = _c_from_inv(c_inv13)
        if ni_p_half is None:
            # p ≤ 0 case: R-based DPI is undefined
            p_val = -1
            n_params = n_obs + 1  # Force p < 0
        else:
            p_val = n_i / ni_p_half ** 2
            n_params = max(0, int(round(n_obs - p_val)))

        params = RefinementParams(
            resolution=d_min,
            r_work=R,
            completeness=comp,
            n_obs=n_obs,
            n_params=n_params,
            n_atoms_refined=n_i,
        )
        atoms = make_atoms(n_i)
        result = calc_r(atoms, params)

        if exp_r is None:
            assert result is None, (
                f"{label}: expected R-based DPI to be None (p ≤ 0) but got {result}"
            )
        else:
            assert result is not None, f"{label}: R-based DPI returned None unexpectedly"
            assert abs(result.sigma_r_avg - exp_r) / exp_r < _REL_TOL, (
                f"{label}: σ(r,R) = {result.sigma_r_avg:.4f}, expected {exp_r}"
            )


# ---------------------------------------------------------------------------
# Formula unit tests
# ---------------------------------------------------------------------------

class TestRFreeFormulaUsesNobs:
    """Critical: R_free formula must use n_obs in the denominator (Cruickshank Eq. 31)."""

    def test_rfree_uses_n_obs_not_n_free(self):
        """
        Verify that changing n_obs changes the result and that n_free has no effect.
        The DPI should scale as sqrt(N_i / n_obs).
        """
        base = dict(resolution=1.0, r_free=0.20, completeness=1.0,
                    n_atoms_refined=1000, n_free=5000)

        # n_obs = 10000
        p1 = RefinementParams(**base, n_obs=10000)
        r1 = calc_rfree(make_atoms(1000), p1)

        # n_obs = 40000 → DPI should be halved (sqrt(1/4))
        p2 = RefinementParams(**base, n_obs=40000)
        r2 = calc_rfree(make_atoms(1000), p2)

        assert r1 is not None and r2 is not None
        ratio = r1.sigma_x_avg / r2.sigma_x_avg
        assert abs(ratio - 2.0) < 0.01, (
            f"Expected ratio 2.0 when n_obs quadrupled; got {ratio:.4f}"
        )

    def test_n_free_has_no_effect_on_rfree_dpi(self):
        """n_free is not used in the R_free formula; changing it must not change the result."""
        base = dict(resolution=1.5, r_free=0.25, completeness=0.95,
                    n_obs=20000, n_atoms_refined=1500)
        p1 = RefinementParams(**base, n_free=1000)
        p2 = RefinementParams(**base, n_free=9000)
        atoms = make_atoms(1500)
        r1 = calc_rfree(atoms, p1)
        r2 = calc_rfree(atoms, p2)
        assert r1 is not None and r2 is not None
        assert r1.sigma_x_avg == pytest.approx(r2.sigma_x_avg), (
            "n_free must not influence the R_free-based DPI"
        )


class TestRBasedFormula:
    """Unit tests for the R-based DPI formula."""

    def test_scales_with_r_factor(self):
        """DPI is proportional to R_work."""
        base = dict(resolution=2.0, completeness=1.0, n_obs=20000, n_params=5000,
                    n_atoms_refined=1000)
        p1 = RefinementParams(**base, r_work=0.20)
        p2 = RefinementParams(**base, r_work=0.40)
        atoms = make_atoms(1000)
        r1 = calc_r(atoms, p1)
        r2 = calc_r(atoms, p2)
        assert r1 is not None and r2 is not None
        ratio = r2.sigma_x_avg / r1.sigma_x_avg
        assert abs(ratio - 2.0) < 0.01

    def test_negative_p_returns_none(self):
        """When N_obs ≤ N_params, R-based DPI must return None."""
        params = RefinementParams(
            resolution=2.5, r_work=0.20, completeness=1.0,
            n_obs=1000, n_params=2000, n_atoms_refined=500,
        )
        result = calc_r(make_atoms(500), params)
        assert result is None

    def test_sigma_r_equals_sqrt3_times_sigma_x(self):
        """σ(r) = √3 × σ(x) always."""
        params = RefinementParams(
            resolution=1.5, r_work=0.18, completeness=0.97,
            n_obs=30000, n_params=8000, n_atoms_refined=1200,
        )
        result = calc_r(make_atoms(1200), params)
        assert result is not None
        assert result.sigma_r_avg == pytest.approx(math.sqrt(3) * result.sigma_x_avg)


# ---------------------------------------------------------------------------
# N_params auto-estimation (Bug #4)
# ---------------------------------------------------------------------------

class TestNParamsAutoEstimation:
    """When N_params is not provided, the calculator should auto-estimate."""

    def test_auto_estimates_when_missing(self, capsys):
        """R-based DPI should succeed even without explicit N_params."""
        n_atoms = 1000
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=50000,  # no n_params provided
            n_atoms_refined=n_atoms,
        )
        atoms = make_atoms(n_atoms)
        result = calc_r(atoms, params)
        # 4 × 1000 = 4000 → p = 50000 - 4000 = 46000 (positive)
        assert result is not None
        captured = capsys.readouterr()
        assert 'auto-estimating' in captured.out.lower() or 'n_params' in captured.out.lower()

    def test_auto_estimate_uses_4x_natoms(self):
        """Estimated N_params = 4 × N_atoms (isotropic B)."""
        n_atoms = 500
        n_obs = 20000
        params = RefinementParams(
            resolution=2.5, r_work=0.20, completeness=1.0,
            n_obs=n_obs, n_atoms_refined=n_atoms,
        )
        atoms = make_atoms(n_atoms)
        result = calc_r(atoms, params)
        assert result is not None
        # p should be n_obs - 4*n_atoms = 20000 - 2000 = 18000
        assert result.p == n_obs - 4 * n_atoms


# ---------------------------------------------------------------------------
# Alternate conformer filtering (Bug #3)
# ---------------------------------------------------------------------------

class TestAltConformerFiltering:
    """Atoms with alt_loc B, C, … must be excluded from the working set."""

    def test_alt_b_excluded(self):
        """Alt-loc B atoms must be excluded from the *working* set (n_atoms_used)
        but must appear in per_atom output so the CSV includes all conformers."""
        atoms = make_atoms(5)
        # Replace two atoms with alt_loc 'B'
        atoms[3].alt_loc = 'B'
        atoms[4].alt_loc = 'B'

        params = RefinementParams(
            resolution=2.0, r_work=0.20, r_free=0.25, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=5,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None
        # Working set: only the 3 primary-conformer atoms
        assert result.n_atoms_used == 3
        # per_atom output: all 5 atoms (3 primary + 2 alt-B) for complete CSV
        assert len(result.per_atom) == 5
        # n_atoms_used must NOT include alt-B atoms
        alt_b_in_output = [ad for ad in result.per_atom if ad.atom.alt_loc == 'B']
        assert len(alt_b_in_output) == 2, (
            "Alt-loc B atoms should appear in per_atom for CSV output"
        )

    def test_alt_a_included(self):
        """Atoms with alt_loc 'A' should be included (primary conformer)."""
        atoms = make_atoms(4)
        atoms[0].alt_loc = 'A'
        atoms[1].alt_loc = 'A'

        params = RefinementParams(
            resolution=2.0, r_work=0.20, r_free=0.25, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=4,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None
        assert result.n_atoms_used == 4


# ---------------------------------------------------------------------------
# Parameter sanity checks (Bug #5)
# ---------------------------------------------------------------------------

class TestParamSanityChecks:
    """Sanity-check warnings are emitted for suspicious parameter values."""

    def test_r_work_greater_than_r_free_warns(self, capsys):
        params = RefinementParams(
            resolution=2.0, r_work=0.35, r_free=0.25, completeness=0.97,
            n_obs=20000, n_params=5000, n_atoms_refined=1000,
        )
        calc = DPICalculator(make_atoms(1000), params, apply_z_correction=False)
        calc.calculate_r_based()
        out = capsys.readouterr().out
        assert 'warning' in out.lower()

    def test_high_r_work_warns(self, capsys):
        params = RefinementParams(
            resolution=2.0, r_work=0.75, r_free=0.80, completeness=0.97,
            n_obs=20000, n_params=5000, n_atoms_refined=1000,
        )
        calc = DPICalculator(make_atoms(1000), params, apply_z_correction=False)
        calc.calculate_r_based()
        out = capsys.readouterr().out
        assert 'warning' in out.lower()

    def test_low_completeness_warns(self, capsys):
        params = RefinementParams(
            resolution=2.0, r_work=0.20, r_free=0.25, completeness=0.40,
            n_obs=20000, n_params=5000, n_atoms_refined=1000,
        )
        calc = DPICalculator(make_atoms(1000), params, apply_z_correction=False)
        calc.calculate_r_based()
        out = capsys.readouterr().out
        assert 'warning' in out.lower()


# ---------------------------------------------------------------------------
# Per-atom B-factor exponent (Bug #2)
# ---------------------------------------------------------------------------

class TestPerAtomBFactorCorrection:
    """Verify per-atom correction uses √(Bᵢ/B_avg) (Helliwell 2023 Eq. 2)."""

    def test_b_factor_scaling(self):
        """
        For an atom with Bᵢ = B_avg + Δ, the per-atom σ(x) should be
        σ(x, B_avg) × √(Bᵢ / B_avg).
        """
        d_min = 2.0
        b_avg = 20.0
        delta_b = 8.0  # second atom has higher B than average

        atoms = [
            Atom(serial=1, name='C', alt_loc='', res_name='ALA', chain_id='A',
                 res_seq=1, ins_code='', x=0, y=0, z=0,
                 occupancy=1.0, b_iso=b_avg, element='C', record_type='ATOM'),
            Atom(serial=2, name='C', alt_loc='', res_name='ALA', chain_id='A',
                 res_seq=2, ins_code='', x=0, y=0, z=0,
                 occupancy=1.0, b_iso=b_avg + delta_b, element='C', record_type='ATOM'),
        ]
        params = RefinementParams(
            resolution=d_min, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=2,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None

        sigma_x_1 = result.per_atom[0].sigma_x
        sigma_x_2 = result.per_atom[1].sigma_x

        # Helliwell (2023) Eq. 2: ratio = √(B2/B1)
        expected_ratio = math.sqrt((b_avg + delta_b) / b_avg)
        actual_ratio = sigma_x_2 / sigma_x_1
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6), (
            f"Expected per-atom ratio {expected_ratio:.6f}, got {actual_ratio:.6f}. "
            "Per-atom correction should use √(Bᵢ/B_avg) per Helliwell (2023)."
        )

    def test_high_b_factor_reasonable(self):
        """Atoms with B=120 vs B_avg=60 should give ~1.41× multiplier, NOT exponential blowup."""
        b_avg = 60.0
        b_high = 120.0

        atoms = [
            Atom(serial=1, name='C', alt_loc='', res_name='ALA', chain_id='A',
                 res_seq=1, ins_code='', x=0, y=0, z=0,
                 occupancy=1.0, b_iso=b_avg, element='C', record_type='ATOM'),
            Atom(serial=2, name='C', alt_loc='', res_name='ALA', chain_id='A',
                 res_seq=2, ins_code='', x=0, y=0, z=0,
                 occupancy=1.0, b_iso=b_high, element='C', record_type='ATOM'),
        ]
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=2,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None

        ratio = result.per_atom[1].sigma_x / result.per_atom[0].sigma_x
        expected = math.sqrt(120.0 / 60.0)  # = 1.414
        assert ratio == pytest.approx(expected, rel=1e-6)
        # Ensure it's NOT the exponential form
        assert ratio < 10.0, f"Per-atom ratio {ratio} is unreasonably large — exponential bug?"

    def test_atom_at_b_avg_has_unit_correction(self):
        """Atom at exactly B_avg should have σ(x,Bᵢ) = σ(x, B_avg)."""
        b_avg = 25.0
        atoms = make_atoms(100, b_iso=b_avg)
        params = RefinementParams(
            resolution=1.5, r_work=0.20, completeness=1.0,
            n_obs=30000, n_params=8000, n_atoms_refined=100,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None
        for ad in result.per_atom:
            assert ad.sigma_x == pytest.approx(result.sigma_x_avg, rel=1e-6)


# ---------------------------------------------------------------------------
# Phenix log parser
# ---------------------------------------------------------------------------

class TestPhenixLogParser:
    """Basic tests for PhenixLogParser."""

    _SAMPLE_LOG = """\
Refinement statistics after last macro-cycle:
  r_work = 0.1923
  r_free = 0.2315
  d_min = 2.10
  completeness (%) = 97.4
  number_of_reflections = 18583
  number_of_test_reflections = 928
  Number of atoms : 1558
"""

    def test_parses_r_work(self, tmp_path):
        f = tmp_path / 'refine.log'
        f.write_text(self._SAMPLE_LOG)
        params = PhenixLogParser.parse(str(f))
        assert params.r_work == pytest.approx(0.1923)

    def test_parses_r_free(self, tmp_path):
        f = tmp_path / 'refine.log'
        f.write_text(self._SAMPLE_LOG)
        params = PhenixLogParser.parse(str(f))
        assert params.r_free == pytest.approx(0.2315)

    def test_parses_resolution(self, tmp_path):
        f = tmp_path / 'refine.log'
        f.write_text(self._SAMPLE_LOG)
        params = PhenixLogParser.parse(str(f))
        assert params.resolution == pytest.approx(2.10)

    def test_parses_completeness(self, tmp_path):
        f = tmp_path / 'refine.log'
        f.write_text(self._SAMPLE_LOG)
        params = PhenixLogParser.parse(str(f))
        assert params.completeness == pytest.approx(0.974)

    def test_parses_n_obs(self, tmp_path):
        f = tmp_path / 'refine.log'
        f.write_text(self._SAMPLE_LOG)
        params = PhenixLogParser.parse(str(f))
        assert params.n_obs == 18583

    def test_parses_n_free(self, tmp_path):
        f = tmp_path / 'refine.log'
        f.write_text(self._SAMPLE_LOG)
        params = PhenixLogParser.parse(str(f))
        assert params.n_free == 928

    def test_parses_n_atoms(self, tmp_path):
        f = tmp_path / 'refine.log'
        f.write_text(self._SAMPLE_LOG)
        params = PhenixLogParser.parse(str(f))
        assert params.n_atoms_refined == 1558


# ---------------------------------------------------------------------------
# Alt-conformer CSV output tests (Issue 1)
# ---------------------------------------------------------------------------

class TestAltConfCSVOutput:
    """Verify that all alternate conformers appear in the per-atom output list.

    The DPI calculation uses only the primary conformer (alt_loc '' or 'A') and
    atoms above min_occupancy for computing B_avg / N_a / Z_avg.  However, the
    per-atom list written to CSV must include *all* conformers so that every atom
    in the structure gets a coordinate-precision estimate.
    """

    def _make_multiconf_atoms(self):
        """Three-conformer residue at occupancy 0.33 each plus a single-conformer residue."""
        atoms = []
        # Residue 1: three conformers (A, B, C) at 0.33 each
        for i, alt in enumerate(('A', 'B', 'C'), start=1):
            atoms.append(Atom(
                serial=i, name='CA', alt_loc=alt, res_name='ALA', chain_id='A',
                res_seq=1, ins_code='', x=0.0, y=0.0, z=0.0,
                occupancy=0.33, b_iso=20.0 + i * 5,  # different B per conformer
                element='C', record_type='ATOM',
            ))
        # Residue 2: single conformer, fully occupied
        atoms.append(Atom(
            serial=4, name='CA', alt_loc='', res_name='GLY', chain_id='A',
            res_seq=2, ins_code='', x=1.0, y=0.0, z=0.0,
            occupancy=1.0, b_iso=30.0,
            element='C', record_type='ATOM',
        ))
        return atoms

    def test_alt_a_appears_in_per_atom(self):
        """Alt-loc A atoms from multi-conformer residues appear in per_atom."""
        atoms = self._make_multiconf_atoms()
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=4,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None
        alt_a = [ad for ad in result.per_atom if ad.atom.alt_loc == 'A']
        assert len(alt_a) >= 1, "Alt-loc A atom must appear in per_atom output"

    def test_alt_b_appears_in_per_atom(self):
        """Alt-loc B atoms from multi-conformer residues appear in per_atom."""
        atoms = self._make_multiconf_atoms()
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=4,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None
        alt_b = [ad for ad in result.per_atom if ad.atom.alt_loc == 'B']
        assert len(alt_b) >= 1, "Alt-loc B atom must appear in per_atom output"

    def test_alt_c_appears_in_per_atom(self):
        """Alt-loc C atoms from multi-conformer residues appear in per_atom."""
        atoms = self._make_multiconf_atoms()
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=4,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None
        alt_c = [ad for ad in result.per_atom if ad.atom.alt_loc == 'C']
        assert len(alt_c) >= 1, "Alt-loc C atom must appear in per_atom output"

    def test_per_atom_uses_individual_b_factor(self):
        """Each conformer gets per-atom DPI scaled by its own √(Bᵢ/B_avg)."""
        atoms = self._make_multiconf_atoms()
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=4,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None

        # Conformers A, B, C have b_iso from _make_multiconf_atoms: 20 + i*5 for i=1,2,3
        expected_b_by_alt = {alt: 20.0 + (i + 1) * 5
                             for i, alt in enumerate(('A', 'B', 'C'))}
        by_alt = {ad.atom.alt_loc: ad for ad in result.per_atom
                  if ad.atom.res_seq == 1}
        assert 'A' in by_alt and 'B' in by_alt and 'C' in by_alt

        # Higher B-factor → larger sigma
        assert by_alt['A'].sigma_x < by_alt['B'].sigma_x < by_alt['C'].sigma_x, (
            "Conformers with larger B-factors should have larger σ(x)"
        )

        # Ratio should match √(Bᵢ/B_avg) for each conformer
        b_avg = result.b_avg
        for alt, expected_b in expected_b_by_alt.items():
            expected_ratio = math.sqrt(expected_b / b_avg)
            actual_ratio = by_alt[alt].sigma_x / result.sigma_x_avg
            assert actual_ratio == pytest.approx(expected_ratio, rel=1e-4), (
                f"Alt-{alt}: ratio {actual_ratio:.4f}, expected √({expected_b}/{b_avg:.2f})={expected_ratio:.4f}"
            )

    def test_n_atoms_used_not_inflated_by_alt_conformers(self):
        """n_atoms_used must count only the working atoms, not all alt conformers."""
        atoms = self._make_multiconf_atoms()
        # Residue 1 has 3 conformers at occ=0.33 each — all below min_occupancy=0.5
        # So working set = only residue 2 (1 atom); per_atom = all 4 atoms
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=4,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False, min_occupancy=0.5)
        result = calc.calculate_r_based()
        assert result is not None

        # Working set excludes the low-occupancy multi-conf atoms
        assert result.n_atoms_used == 1, (
            f"n_atoms_used should be 1 (only full-occupancy residue 2); got {result.n_atoms_used}"
        )
        # per_atom includes ALL 4 atoms (all conformers, all occupancies)
        assert len(result.per_atom) == 4, (
            f"per_atom should have 4 entries (all atoms including alt-conf); got {len(result.per_atom)}"
        )

    def test_low_occupancy_atoms_in_per_atom(self):
        """Low-occupancy atoms must appear in per_atom even if excluded from working set."""
        # Single atom at occupancy 0.2 (below min_occupancy)
        low_occ_atom = Atom(
            serial=1, name='CA', alt_loc='', res_name='ALA', chain_id='A',
            res_seq=1, ins_code='', x=0.0, y=0.0, z=0.0,
            occupancy=0.2, b_iso=25.0, element='C', record_type='ATOM',
        )
        full_occ_atom = Atom(
            serial=2, name='CA', alt_loc='', res_name='GLY', chain_id='A',
            res_seq=2, ins_code='', x=1.0, y=0.0, z=0.0,
            occupancy=1.0, b_iso=25.0, element='C', record_type='ATOM',
        )
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=2,
        )
        calc = DPICalculator([low_occ_atom, full_occ_atom], params,
                             apply_z_correction=False, min_occupancy=0.5)
        result = calc.calculate_r_based()
        assert result is not None
        assert result.n_atoms_used == 1, "Only full-occupancy atom in working set"
        assert len(result.per_atom) == 2, "Both atoms in per_atom output"
        serials = [ad.atom.serial for ad in result.per_atom]
        assert 1 in serials, "Low-occupancy atom (serial=1) must appear in per_atom"

    def test_csv_writer_includes_all_conformers(self, tmp_path):
        """write_csv() must output a row for every conformer."""
        from dpi_calculator import write_csv

        atoms = self._make_multiconf_atoms()
        params = RefinementParams(
            resolution=2.0, r_work=0.20, completeness=1.0,
            n_obs=20000, n_params=5000, n_atoms_refined=4,
        )
        calc = DPICalculator(atoms, params, apply_z_correction=False)
        result = calc.calculate_r_based()
        assert result is not None

        csv_path = str(tmp_path / 'dpi.csv')
        write_csv(result, csv_path)

        with open(csv_path) as fh:
            rows = list(fh)
        # 1 header + 4 data rows (3 conformers of residue 1 + 1 for residue 2)
        assert len(rows) == 5, (
            f"CSV should have 5 rows (header + 4 atoms); got {len(rows)}"
        )
        # All three alt_loc values should appear
        content = ''.join(rows)
        for alt in ('A', 'B', 'C'):
            assert f',{alt},' in content, f"Alt-loc {alt} not found in CSV output"


# ---------------------------------------------------------------------------
# End-to-end tests — download actual PDB files from RCSB (Tier 2)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestTable3EndToEnd:
    """End-to-end validation: download PDB files from RCSB and run the full pipeline.

    These tests exercise the parsers *and* the DPI formulas against the actual
    deposited structures corresponding to Cruickshank (1999) Table 3.  They
    require network access and are skipped by default; run with::

        pytest --run-network test_dpi_calculator.py::TestTable3EndToEnd

    PDB ID Mapping — user-researched and confirmed
    -----------------------------------------------
    | Protein               | PDB ID | Reference                 | Status           |
    |-----------------------|--------|---------------------------|------------------|
    | Concanavalin A        | 1NLS   | Deacon et al. (1997)      | ✅ should match  |
    | HEW lysozyme (ground) | 193L   | Vaney et al. (1996)       | ✅ should match  |
    | HEW lysozyme (space)  | 194L   | Vaney et al. (1996)       | ✅ should match  |
    | γB-crystallin         | 1GCS   | Tickle et al. (1998a)     | ✅ should match  |
    | βB2-crystallin        | 2BB2   | Tickle et al. (1998a)     | ✅ should match  |
    | β-purothionin         | 1BHP   | Stec, Rao et al. (1995)   | ✅ should match  |
    | α₁-purothionin        | 2PLH   | Rao et al. (1995)         | ✅ should match  |
    | EM lysozyme           | 1JUG   | Guss et al. (1997)        | ✅ should match  |
    | Azurin II             | 1ARN   | Dodd et al. (1995)        | ⚠️ known mismatch|
    | Ribonuclease A+RI     | 1DFJ   | Kobe & Deisenhofer (1995) | ✅ should match  |
    | Fab HyHEL-5 + HEWL   | 3HFL   | Cohen et al. (1996)       | ⚠️ OBSOLETE      |
    | Immunoglobulin        | 1BWW   | Usón et al. (1999)        | ⚠️ Table 1 only  |

    Notes on HETATM / metal ions
    -----------------------------
    Metal ions (Cu, Fe, Zn, …) and other non-standard ligands are stored as
    HETATM records in PDB/mmCIF files.  The DPICalculator excludes HETATM by
    default (``include_hetatm=False``).  For structures with catalytic metals
    (e.g. Azurin II which has Cu²⁺) use ``include_hetatm=True`` to include the
    metal in the atom count, which more closely matches what Cruickshank used.
    The end-to-end test for 1ARN demonstrates this flag.
    """

    # Wider tolerance for end-to-end tests: PDB depositions may have been
    # updated since Cruickshank (1999) and parsing may differ slightly.
    # 15% tolerance accommodates known small discrepancies (e.g. 1NLS gives
    # σ(r,R)=0.031 vs the paper's 0.034, a 9% gap).
    _E2E_TOL = 0.15  # 15 %

    # (pdb_id, label, exp_d_min, exp_rfree_dpi, exp_r_dpi_or_None, has_deposited_rfree)
    #
    # has_deposited_rfree=False: older depositions (pre-2000) that do not have
    # R_free stored in the mmCIF _refine.ls_r_factor_r_free field.  The R_free
    # DPI check is skipped gracefully for these structures; R-based DPI is tested
    # instead where exp_r_dpi_or_None is not None.
    #
    # 1GCS (γB-crystallin) is moved to a dedicated xfail test below because the
    # deposited resolution (2.000 Å) differs substantially from the 1.49 Å used
    # by Cruickshank from the Tickle et al. (1998a) manuscript.
    _STRUCTURES = [
        ('1NLS', 'Concanavalin A',     0.94, 0.036, 0.034, True),
        # 1NLS: deposited values yield σ(r,R)=0.031 vs paper's 0.034 (9% gap);
        # within the 15% e2e tolerance.
        ('193L', 'HEW lys ground',     1.33, 0.12,  0.11,  True),
        ('194L', 'HEW lys space',      1.40, 0.13,  0.12,  True),
        ('2BB2', 'betaB2-crystallin',  2.10, 0.22,  0.25,  False),
        # 2BB2: R_free not stored in deposited mmCIF (pre-2000 deposition).
        ('1BHP', 'beta-purothionin',   1.70, 0.26,  0.22,  True),
        ('2PLH', 'alpha1-purothionin', 2.50, 0.68,  None,  False),
        # 2PLH: R_free not stored in deposited mmCIF; R-based DPI undefined (p≤0).
        ('1JUG', 'EM lysozyme',        1.90, 0.28,  0.30,  True),
        ('1DFJ', 'RNase A+RI',         2.50, 0.69,  1.85,  False),
        # 1DFJ: R_free not stored in deposited mmCIF (pre-2000 deposition).
    ]

    @pytest.mark.parametrize(
        'pdb_id,label,exp_d_min,exp_rfree_dpi,exp_r_dpi,has_deposited_rfree',
        _STRUCTURES,
        ids=[s[0] for s in _STRUCTURES],
    )
    def test_download_parse_and_dpi(
        self, tmp_path, pdb_id, label, exp_d_min, exp_rfree_dpi, exp_r_dpi,
        has_deposited_rfree,
    ):
        """Download mmCIF from RCSB, parse, compute DPI, compare against Table 3."""
        from dpi_calculator import download_pdb, GemmiParser, DPICalculator

        filepath = download_pdb(pdb_id, dest_dir=str(tmp_path), prefer_mmcif=True)
        atoms, params = GemmiParser.parse(filepath)

        assert atoms, f"{pdb_id}: no atoms parsed from {filepath}"
        assert params.resolution is not None, f"{pdb_id}: resolution not parsed"
        assert abs(params.resolution - exp_d_min) / exp_d_min < self._E2E_TOL, (
            f"{pdb_id}: parsed resolution {params.resolution:.3f} Å, "
            f"expected {exp_d_min} Å ± {self._E2E_TOL * 100:.0f}%"
        )

        calc = DPICalculator(atoms, params, include_hetatm=False, apply_z_correction=False)

        # R_free-based DPI — check only when R_free is expected in the deposited file
        rfree_result = calc.calculate_rfree_based()
        if has_deposited_rfree:
            assert rfree_result is not None, (
                f"{pdb_id}: R_free-based DPI returned None "
                f"(parsed r_free={params.r_free}, n_obs={params.n_obs})"
            )
            assert abs(rfree_result.sigma_r_avg - exp_rfree_dpi) / exp_rfree_dpi < self._E2E_TOL, (
                f"{pdb_id} ({label}): σ(r,Rfree) = {rfree_result.sigma_r_avg:.3f} Å, "
                f"expected {exp_rfree_dpi} Å ± {self._E2E_TOL * 100:.0f}%"
            )
        else:
            # Older deposition without R_free in mmCIF — skip R_free check
            if rfree_result is not None:
                # Opportunistically validate if R_free was somehow available
                assert abs(rfree_result.sigma_r_avg - exp_rfree_dpi) / exp_rfree_dpi < self._E2E_TOL, (
                    f"{pdb_id} ({label}): σ(r,Rfree) = {rfree_result.sigma_r_avg:.3f} Å, "
                    f"expected {exp_rfree_dpi} Å ± {self._E2E_TOL * 100:.0f}%"
                )

        # R-based DPI — only check when an expected value is provided
        if exp_r_dpi is not None:
            r_result = calc.calculate_r_based()
            assert r_result is not None, (
                f"{pdb_id}: R-based DPI returned None unexpectedly "
                f"(parsed n_obs={params.n_obs}, n_params={params.n_params})"
            )
            assert abs(r_result.sigma_r_avg - exp_r_dpi) / exp_r_dpi < self._E2E_TOL, (
                f"{pdb_id} ({label}): σ(r,R) = {r_result.sigma_r_avg:.3f} Å, "
                f"expected {exp_r_dpi} Å ± {self._E2E_TOL * 100:.0f}%"
            )

    @pytest.mark.xfail(
        reason=(
            "1GCS (γB-crystallin): The deposited structure has resolution 2.000 Å, "
            "which differs substantially from the 1.49 Å used by Cruickshank from "
            "the Tickle et al. (1998a) manuscript.  The deposited values do not match "
            "the Table 3 entries and cannot be compared directly."
        ),
        strict=False,
    )
    def test_gammab_crystallin_1GCS_known_mismatch(self, tmp_path):
        """γB-crystallin (1GCS): document known resolution mismatch vs Cruickshank Table 3."""
        from dpi_calculator import download_pdb, GemmiParser, DPICalculator

        filepath = download_pdb('1GCS', dest_dir=str(tmp_path), prefer_mmcif=True)
        atoms, params = GemmiParser.parse(filepath)

        exp_d_min = 1.49
        assert params.resolution is not None
        assert abs(params.resolution - exp_d_min) / exp_d_min < self._E2E_TOL, (
            f"1GCS: parsed resolution {params.resolution:.3f} Å, expected {exp_d_min} Å"
        )

    @pytest.mark.xfail(
        reason=(
            "1ARN (Azurin II): The PDB was deposited in 2000, after Cruickshank (1999) "
            "was published.  Cruickshank used Dodd et al. (1995) values "
            "(R=0.188, R_free=0.207, n_obs=12162) which differ from the deposited entry. "
            "The structure may have been re-refined; parsed values will not match Table 3."
        ),
        strict=False,
    )
    def test_azurin_ii_1ARN_known_mismatch(self, tmp_path):
        """Azurin II (1ARN): document known mismatch between PDB and Cruickshank values."""
        from dpi_calculator import download_pdb, GemmiParser, DPICalculator

        filepath = download_pdb('1ARN', dest_dir=str(tmp_path), prefer_mmcif=True)
        atoms, params = GemmiParser.parse(filepath)

        calc = DPICalculator(atoms, params, include_hetatm=False, apply_z_correction=False)
        rfree_result = calc.calculate_rfree_based()
        assert rfree_result is not None

        # Table 3 expected: 0.23 Å; deposited values differ
        exp_rfree_dpi = 0.23
        assert abs(rfree_result.sigma_r_avg - exp_rfree_dpi) / exp_rfree_dpi < self._E2E_TOL, (
            f"1ARN: σ(r,Rfree) = {rfree_result.sigma_r_avg:.3f}, expected {exp_rfree_dpi}"
        )

    def test_azurin_ii_1ARN_include_hetatm(self, tmp_path):
        """Azurin II (1ARN) has Cu²⁺ (HETATM); verify include_hetatm flag works."""
        from dpi_calculator import download_pdb, GemmiParser, DPICalculator

        filepath = download_pdb('1ARN', dest_dir=str(tmp_path), prefer_mmcif=True)
        atoms, params = GemmiParser.parse(filepath)

        calc_with = DPICalculator(atoms, params, include_hetatm=True,  apply_z_correction=False)
        calc_without = DPICalculator(atoms, params, include_hetatm=False, apply_z_correction=False)

        res_with    = calc_with.calculate_rfree_based()
        res_without = calc_without.calculate_rfree_based()

        assert res_with is not None and res_without is not None
        # Including HETATM should increase (or equal) the atom count
        assert res_with.n_atoms_used >= res_without.n_atoms_used, (
            "include_hetatm=True should result in >= atoms compared to False; "
            f"got {res_with.n_atoms_used} vs {res_without.n_atoms_used}"
        )

    def test_fab_hyhel5_3HFL_obsolete(self):
        """3HFL (Fab HyHEL-5 with HEWL) is OBSOLETE on RCSB — cannot be downloaded.

        This PDB entry was replaced by a higher-resolution structure.  The end-to-end
        test for 3HFL is permanently skipped with this documented note.  The
        formula-only test (TestTable3) still validates the calculation using the
        Cruickshank Table 3 intermediate values directly.

        Note on Rfree DPI discrepancy: Table 3 prints 0.69 for Rfree DPI, but
        computing from the tabulated intermediates gives ~0.89.  The printed value
        appears to be a transcription error (same as RNase A row above it).  The
        formula test uses 0.89.
        """
        pytest.skip(
            "3HFL (Fab HyHEL-5 with HEWL) is OBSOLETE on RCSB and cannot be "
            "downloaded; it has been superseded by a higher-resolution structure."
        )

    @pytest.mark.xfail(
        reason=(
            "1BWW (Immunoglobulin): This structure is from Cruickshank Table 1 only "
            "(not Table 3).  The Usón et al. manuscript was 'in preparation' when "
            "Cruickshank published; the deposited model has better R-factors than the "
            "values used in the paper (R=0.156, σ(r)=0.221 Å).  Known mismatch."
        ),
        strict=False,
    )
    def test_immunoglobulin_1BWW_table1_mismatch(self, tmp_path):
        """1BWW (Immunoglobulin, Table 1 only): document known mismatch."""
        from dpi_calculator import download_pdb, GemmiParser, DPICalculator

        filepath = download_pdb('1BWW', dest_dir=str(tmp_path), prefer_mmcif=True)
        atoms, params = GemmiParser.parse(filepath)

        calc = DPICalculator(atoms, params, include_hetatm=False, apply_z_correction=False)
        r_result = calc.calculate_r_based()
        assert r_result is not None

        # Table 1 expected: σ(r,R) = 0.221 Å
        exp_r_dpi = 0.221
        assert abs(r_result.sigma_r_avg - exp_r_dpi) / exp_r_dpi < self._E2E_TOL, (
            f"1BWW: σ(r,R) = {r_result.sigma_r_avg:.3f}, expected {exp_r_dpi}"
        )
