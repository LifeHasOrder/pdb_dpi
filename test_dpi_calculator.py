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
    # Concanavalin A — Deacon et al. (1997)
    ('Concanavalin A',     2130, 116712, 0.148, 1.099, 0.128, 0.148, 0.94,  0.034, 0.036),
    # HEW lysozyme ground state — Vaney et al. (1996)
    ('HEW lys ground',     1145,  24111, 0.242, 1.048, 0.184, 0.226, 1.33,  0.11,  0.12),
    # HEW lysozyme space state — Vaney et al. (1996)
    ('HEW lys space',      1141,  21542, 0.259, 1.040, 0.183, 0.226, 1.40,  0.12,  0.13),
    # γB-crystallin — Tickle et al. (1998a)
    ('gammaB-crystallin',  1708,  26151, 0.297, 1.032, 0.180, 0.204, 1.49,  0.14,  0.14),
    # βB2-crystallin — Tickle et al. (1998a)
    ('betaB2-crystallin',  1558,  18583, 0.356, 1.032, 0.184, 0.200, 2.10,  0.25,  0.22),
    # β-purothionin — Stec, Rao et al. (1995)
    ('beta-purothionin',    439,   4966, 0.370, 1.050, 0.198, 0.281, 1.70,  0.22,  0.26),
    # EM lysozyme — Guss et al. (1997)
    ('EM lysozyme',        1068,   8308, 0.514, 1.040, 0.169, 0.229, 1.90,  0.30,  0.28),
    # Azurin II — Dodd et al. (1995)
    ('Azurin II',          1012,  12162, 0.353, 1.174, 0.188, 0.207, 1.90,  0.26,  0.23),
    # Ribonuclease A with RI — Kobe & Deisenhofer (1995)
    ('RNase A+RI',         4416,  18859, 1.922, 1.145, 0.194, 0.286, 2.50,  1.85,  0.69),
    # α₁-purothionin — Rao et al. (1995)  [p < 0 for R-based; R-based is undefined]
    ('alpha1-purothionin',  434,   1168,  None, 1.180, 0.155, 0.218, 2.50,  None,  0.68),
    # Fab HyHEL-5 with HEWL — Cohen et al. (1996)  [p < 0 for R-based]
    # Note: The problem statement lists 0.69 for R_free DPI here, which appears to be
    # a transcription error (duplicated from the RNase A row above).  Computing from
    # the tabulated intermediate values ((Ni/n_obs)^½=0.607, C^-1/3=1.111,
    # R_free=0.288, d_min=2.65) gives σ(r) ≈ 0.89.
    ('Fab HyHEL5+HEWL',   4333,  11754,  None, 1.111, 0.196, 0.288, 2.65,  None,  0.89),
]

# Tolerance: published values are rounded to 2 significant figures; allow 15%
_REL_TOL = 0.15


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
        """Atoms with alt_loc 'B' must not appear in per-atom results."""
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
        # Only 3 atoms should be in per-atom list
        assert result.n_atoms_used == 3
        assert all(ad.atom.alt_loc in ('', 'A') for ad in result.per_atom)

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

class TestPerAtomBFactorExponent:
    """Verify per-atom correction uses 2×d_min² denominator (Gurusaran 2014 Eq. 6)."""

    def test_b_factor_scaling(self):
        """
        For an atom with B_i = B_avg + Δ, the per-atom σ(x) should be
        σ(x, B_avg) × exp(Δ / (2 × d_min²)).
        """
        d_min = 2.0
        b_avg = 20.0
        delta_b = 8.0  # atom has higher B than average

        # Two atoms: one at b_avg, one at b_avg + delta_b
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

        sigma_x_avg = result.sigma_x_avg
        # atom at b_avg: correction = exp(0) = 1
        sigma_x_1 = result.per_atom[0].sigma_x
        # atom at b_avg + delta_b: correction = exp(delta_b / (2 * d_min^2))
        sigma_x_2 = result.per_atom[1].sigma_x

        expected_ratio = math.exp(delta_b / (2.0 * d_min ** 2))
        actual_ratio = sigma_x_2 / sigma_x_1
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6), (
            f"Expected per-atom ratio {expected_ratio:.6f}, got {actual_ratio:.6f}. "
            "Check that denominator is 2×d_min² not 4×d_min²."
        )

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
