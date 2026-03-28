#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPI Calculator – Diffraction Precision Index for protein crystal structures
===========================================================================

Implements Cruickshank (1999) DPI with per-atom extension (Gurusaran et al. 2014,
Kumar et al. 2015), reproducing the calculations of the now-offline Online_DPI server.

Theory
------
Overall DPI (x-coordinate precision at average B):
    σ(x, B_avg) = sqrt(N_a / p)     × C^(-1/3) × R      × d_min  [R-based]
    σ(x, B_avg) = sqrt(N_a / n_obs) × C^(-1/3) × R_free × d_min  [R_free-based]

Isotropic position error:
    σ(r, B_avg) = sqrt(3) × σ(x, B_avg)

Per-atom coordinate precision (Helliwell 2023, Eq. 2):
    σ(x, B_i) = σ(x, B_avg) × (Z_avg / Z_i) × √(B_i / B_avg)
    σ(r, B_i) = sqrt(3) × σ(x, B_i)

where:
    N_a    = number of non-H atoms included in refinement
    p      = N_obs − N_params  (degrees of freedom; can be negative at low resolution)
    n_obs  = number of observed reflections used in refinement (used in R_free formula)
    N_params = number of refined parameters
    C      = data completeness (0–1)
    R      = conventional R-factor (R_work)
    R_free = free R-factor
    d_min  = maximum resolution (Å)
    B_avg  = average B-factor of all non-H atoms
    B_i    = B-factor of atom i
    Z_i    = atomic number of atom i
    Z_avg  = scattering-weighted average atomic number

Note on the R_free formula: Cruickshank (1999) Eq. 31 and Table 3 clearly show that
the R_free formula uses n_obs (total observed reflections) as the denominator, NOT
N_free (number of free-set reflections).  The factor (N_i/n_obs)^½ is tabulated
explicitly in Table 3 for every protein in that column.

References
----------
Cruickshank, D. W. J. (1999). Acta Cryst. D55, 583-601.
Blow, D. M. (2002). Acta Cryst. D58, 792-797.
Gurusaran, M. et al. (2014). IUCrJ 1, 74-81.
Kumar, K. S. D. et al. (2015). J. Appl. Cryst. 48, 939-942.
Helliwell, J. R. (2023). Curr. Res. Struct. Biol. 6, 100111.
"""

import sys
import math
import argparse
import os
import re
import urllib.request
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ---------------------------------------------------------------------------
# Atomic numbers for elements commonly found in macromolecular structures
# ---------------------------------------------------------------------------
ATOMIC_NUMBERS: Dict[str, int] = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15,
    'S': 16, 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20, 'SC': 21, 'TI': 22,
    'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29,
    'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36,
    'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43,
    'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50,
    'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57,
    'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64,
    'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71,
    'HF': 72, 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78,
    'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85,
    'RN': 86, 'FR': 87, 'RA': 88, 'AC': 89, 'TH': 90, 'PA': 91, 'U': 92,
}


def get_atomic_number(element: str) -> int:
    """Return atomic number for element symbol, defaulting to 7 (N) if unknown."""
    return ATOMIC_NUMBERS.get(element.strip().upper(), 7)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    serial: int
    name: str
    alt_loc: str
    res_name: str
    chain_id: str
    res_seq: int
    ins_code: str
    x: float
    y: float
    z: float
    occupancy: float
    b_iso: float
    element: str
    record_type: str  # ATOM or HETATM

    @property
    def atomic_number(self) -> int:
        return get_atomic_number(self.element)

    @property
    def is_hydrogen(self) -> bool:
        return self.element.strip().upper() in ('H', 'D')

    @property
    def is_fully_occupied(self) -> bool:
        return self.occupancy >= 0.99


@dataclass
class RefinementParams:
    """All crystallographic parameters needed for DPI calculation."""
    resolution: Optional[float] = None       # d_min in Å
    r_work: Optional[float] = None           # R-factor (R_work)
    r_free: Optional[float] = None           # Free R-factor
    completeness: Optional[float] = None     # Data completeness (0–1)
    n_obs: Optional[int] = None              # Number of observed reflections used
    n_params: Optional[int] = None           # Number of refined parameters
    n_free: Optional[int] = None             # Number of free reflections
    n_atoms_refined: Optional[int] = None    # Number of atoms in refinement
    space_group: str = ''
    source: str = ''                         # Where params came from

    def summary(self) -> str:
        lines = [f"Refinement Parameters (source: {self.source or 'unknown'})"]
        lines.append(f"  Resolution (d_min):  {self.resolution} Å")
        lines.append(f"  R_work:              {self.r_work:.4f}" if self.r_work else "  R_work:              N/A")
        lines.append(f"  R_free:              {self.r_free:.4f}" if self.r_free else "  R_free:              N/A")
        lines.append(f"  Completeness:        {self.completeness:.4f}" if self.completeness else "  Completeness:        N/A")
        lines.append(f"  N_obs (reflections): {self.n_obs}")
        lines.append(f"  N_params:            {self.n_params}")
        lines.append(f"  N_free reflections:  {self.n_free}")
        lines.append(f"  N_atoms (refined):   {self.n_atoms_refined}")
        return '\n'.join(lines)


@dataclass
class AtomDPI:
    atom: Atom
    sigma_x: float      # x-coordinate precision (Å)
    sigma_r: float      # isotropic position error (Å)


@dataclass
class DPIResult:
    params: RefinementParams
    method: str                  # 'R' or 'Rfree'
    sigma_x_avg: float           # overall σ(x, B_avg)
    sigma_r_avg: float           # overall σ(r, B_avg) = sqrt(3)*sigma_x_avg
    p: Optional[int]             # N_obs - N_params
    b_avg: float                 # mean B-factor used
    z_avg: float                 # scattering-weighted mean Z
    n_atoms_used: int            # atoms used in per-atom calc
    per_atom: List[AtomDPI] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDB parser (REMARK 3 + ATOM/HETATM)
# ---------------------------------------------------------------------------

class PDBParser:
    """Parse PDB flat files for atoms and REMARK 3 refinement statistics."""

    # Patterns for REMARK 3 fields
    _REMARK3_PATTERNS = {
        'resolution': [
            r'RESOLUTION RANGE HIGH \(ANGSTROMS\)\s*:\s*([\d.]+)',
            r'RESOLUTION\s*[:(]\s*([\d.]+)',
            r'HIGHEST RESOLUTION SHELL.*?:\s*([\d.]+)',
        ],
        'r_work': [
            r'R VALUE\s+\(WORKING SET\)\s*:\s*([\d.]+)',
            r'R VALUE\s+\(WORKING\)\s*:\s*([\d.]+)',
            r'^REMARK   3   R VALUE\s*\(WORKING\)\s*:\s*([\d.]+)',
            r'R VALUE\s+\(WORKING SET,\s*NO CUTOFF\)\s*:\s*([\d.]+)',
        ],
        'r_free': [
            r'FREE R VALUE\s*:\s*([\d.]+)',
            r'R FREE\s*:\s*([\d.]+)',
        ],
        'completeness': [
            r'COMPLETENESS FOR RANGE\s*\(%\)\s*:\s*([\d.]+)',
            r'COMPLETENESS\s*\(%\)\s*:\s*([\d.]+)',
            r'COMPLETENESS\s*:\s*([\d.]+)',
        ],
        'n_obs': [
            r'NUMBER OF REFLECTIONS\s*:\s*(\d+)',
            r'REFLECTIONS USED IN REFINEMENT\s*:\s*(\d+)',
            r'USED IN REFINEMENT\s*:\s*(\d+)',
        ],
        'n_free': [
            r'FREE R VALUE TEST SET SIZE\s*\(%\)\s*:\s*([\d.]+)',  # % — handle separately
            r'NUMBER OF FREE REFLECTIONS\s*:\s*(\d+)',
            r'FREE R SET COUNT\s*:\s*(\d+)',
            r'FREE R VALUE TEST SET COUNT\s*:\s*(\d+)',
        ],
        'n_params': [
            r'NUMBER OF PARAMETERS IN REFINEMENT\s*:\s*(\d+)',
            r'PARAMETERS\s*:\s*(\d+)',
        ],
        'n_atoms': [
            r'TOTAL NUMBER OF ATOMS\s*:\s*(\d+)',
            r'PROTEIN ATOMS\s*:\s*(\d+)',
        ],
        'r_free_pct': [
            r'FREE R VALUE TEST SET SIZE\s*\(%\)\s*:\s*([\d.]+)',
        ],
    }

    @classmethod
    def parse(cls, filepath: str) -> Tuple[List[Atom], RefinementParams]:
        atoms = []
        params = RefinementParams(source='PDB REMARK 3')
        remark3_lines = []

        with open(filepath, 'r', errors='replace') as fh:
            for line in fh:
                rec = line[:6].strip()
                if rec == 'REMARK' and line[6:10].strip() == '3':
                    remark3_lines.append(line[10:].rstrip())
                elif rec in ('ATOM', 'HETATM'):
                    atom = cls._parse_atom_line(line.rstrip('\n'))
                    if atom:
                        atoms.append(atom)

        cls._parse_remark3(remark3_lines, params)
        return atoms, params

    @classmethod
    def _parse_atom_line(cls, line: str) -> Optional[Atom]:
        try:
            record_type = line[0:6].strip()
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            alt_loc = line[16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq_str = line[22:26].strip()
            res_seq = int(res_seq_str) if res_seq_str else 0
            ins_code = line[26].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            occ_str = line[54:60].strip()
            occupancy = float(occ_str) if occ_str else 1.0
            b_str = line[60:66].strip()
            b_iso = float(b_str) if b_str else 0.0
            # Element column (76:78) preferred, fallback to atom name
            element = line[76:78].strip() if len(line) > 76 else ''
            if not element:
                # Infer from atom name: strip leading digits/spaces, take alpha prefix
                raw = name.strip()
                # Remove leading digits
                raw = re.sub(r'^\d+', '', raw)
                # Common 2-letter elements in macromolecules
                two = raw[:2].upper()
                if two in ATOMIC_NUMBERS:
                    element = two
                else:
                    element = raw[:1].upper() if raw else 'C'
            return Atom(
                serial=serial, name=name, alt_loc=alt_loc, res_name=res_name,
                chain_id=chain_id, res_seq=res_seq, ins_code=ins_code,
                x=x, y=y, z=z, occupancy=occupancy, b_iso=b_iso,
                element=element.upper(), record_type=record_type,
            )
        except (ValueError, IndexError):
            return None

    @classmethod
    def _parse_remark3(cls, lines: List[str], params: RefinementParams):
        text = '\n'.join(lines)

        def find(patterns):
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
                if m:
                    return m.group(1)
            return None

        res = find(cls._REMARK3_PATTERNS['resolution'])
        if res:
            params.resolution = float(res)

        r_work = find(cls._REMARK3_PATTERNS['r_work'])
        if r_work:
            v = float(r_work)
            params.r_work = v if v < 1 else v / 100.0

        r_free = find(cls._REMARK3_PATTERNS['r_free'])
        if r_free:
            v = float(r_free)
            params.r_free = v if v < 1 else v / 100.0

        comp = find(cls._REMARK3_PATTERNS['completeness'])
        if comp:
            v = float(comp)
            params.completeness = v / 100.0 if v > 1 else v

        n_obs = find(cls._REMARK3_PATTERNS['n_obs'])
        if n_obs:
            params.n_obs = int(n_obs)

        n_params = find(cls._REMARK3_PATTERNS['n_params'])
        if n_params:
            params.n_params = int(n_params)

        # n_free: look for explicit count first, then % of n_obs
        n_free_direct = None
        for pat in [
            r'FREE R VALUE TEST SET COUNT\s*:\s*(\d+)',
            r'NUMBER OF FREE REFLECTIONS\s*:\s*(\d+)',
            r'FREE R SET COUNT\s*:\s*(\d+)',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                n_free_direct = int(m.group(1))
                break
        if n_free_direct:
            params.n_free = n_free_direct
        else:
            pct_m = re.search(
                r'FREE R VALUE TEST SET SIZE\s*\(%\)\s*:\s*([\d.]+)', text, re.IGNORECASE)
            if pct_m and params.n_obs:
                params.n_free = int(round(float(pct_m.group(1)) / 100.0 * params.n_obs))

        n_atoms = find(cls._REMARK3_PATTERNS['n_atoms'])
        if n_atoms:
            params.n_atoms_refined = int(n_atoms)


# ---------------------------------------------------------------------------
# mmCIF parser
# ---------------------------------------------------------------------------

class MMCIFParser:
    """Parse mmCIF files for atoms and refinement statistics."""

    @classmethod
    def parse(cls, filepath: str) -> Tuple[List[Atom], RefinementParams]:
        with open(filepath, 'r', errors='replace') as fh:
            content = fh.read()

        params = RefinementParams(source='mmCIF _refine/_reflns')
        atoms = []

        # Extract scalar items first
        scalar_map = cls._extract_scalars(content)
        cls._fill_params(scalar_map, params)

        # Extract atom_site loop
        atoms = cls._extract_atoms(content)

        return atoms, params

    @classmethod
    def _extract_scalars(cls, content: str) -> Dict[str, str]:
        """Extract single-value data items from mmCIF."""
        result = {}
        # Match  _category.item   value  (value may be quoted or bare)
        pattern = re.compile(
            r'(_[\w.]+)\s+([\'"])(.*?)\2|(_[\w.]+)\s+(\S+)',
            re.DOTALL
        )
        for m in pattern.finditer(content):
            if m.group(1):
                result[m.group(1).lower()] = m.group(3)
            else:
                result[m.group(4).lower()] = m.group(5)
        return result

    @classmethod
    def _fill_params(cls, d: Dict[str, str], params: RefinementParams):
        def fget(keys):
            for k in keys:
                v = d.get(k)
                if v and v not in ('.', '?'):
                    try:
                        return float(v)
                    except ValueError:
                        pass
            return None

        def iget(keys):
            v = fget(keys)
            return int(v) if v is not None else None

        params.resolution = fget([
            '_refine.ls_d_res_high',
            '_reflns.d_resolution_high',
            '_reflns_shell.d_res_high',
        ])
        params.r_work = fget([
            '_refine.ls_r_factor_r_work',
            '_refine.ls_r_factor_obs',
            '_refine.ls_r_factor_all',
        ])
        params.r_free = fget([
            '_refine.ls_r_factor_r_free',
        ])
        comp = fget([
            '_reflns.percent_possible_all',
            '_reflns.pdbx_percent_possible_all',
            '_reflns.pdbx_percent_possible_obs',
        ])
        if comp is not None:
            params.completeness = comp / 100.0 if comp > 1 else comp

        params.n_obs = iget([
            '_refine.ls_number_reflns_obs',
            '_refine.ls_number_reflns_all',
        ])
        params.n_free = iget([
            '_refine.ls_number_reflns_r_free',
        ])
        params.n_params = iget([
            '_refine.ls_number_parameters',
        ])
        params.n_atoms_refined = iget([
            '_refine.ls_number_atoms_total',
            '_refine_hist.number_atoms_total',
        ])

        sg = d.get('_symmetry.space_group_name_h-m') or d.get('_space_group.name_h-m_alt', '')
        params.space_group = sg.strip("'\" ")

    @classmethod
    def _extract_atoms(cls, content: str) -> List[Atom]:
        """Extract atom_site loop from mmCIF content."""
        atoms = []
        # Find the atom_site loop
        loop_pattern = re.compile(
            r'loop_\s*((?:_atom_site\.\S+\s*)+)(.*?)(?=loop_|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        m = loop_pattern.search(content)
        if not m:
            return atoms

        header_block = m.group(1)
        data_block = m.group(2).strip()

        # Parse column names
        cols = [c.strip().lower() for c in re.findall(r'_atom_site\.\S+', header_block)]
        if not cols:
            return atoms

        col_idx = {c: i for i, c in enumerate(cols)}

        def cidx(names):
            for n in names:
                if n in col_idx:
                    return col_idx[n]
            return None

        ix = cidx(['_atom_site.id'])
        igroup = cidx(['_atom_site.group_pdb'])
        iname = cidx(['_atom_site.label_atom_id', '_atom_site.auth_atom_id'])
        ialt = cidx(['_atom_site.label_alt_id'])
        iresname = cidx(['_atom_site.label_comp_id', '_atom_site.auth_comp_id'])
        ichain = cidx(['_atom_site.label_asym_id', '_atom_site.auth_asym_id'])
        iseq = cidx(['_atom_site.label_seq_id', '_atom_site.auth_seq_id'])
        iins = cidx(['_atom_site.pdbx_pdb_ins_code'])
        ixx = cidx(['_atom_site.cartn_x'])
        iyy = cidx(['_atom_site.cartn_y'])
        izz = cidx(['_atom_site.cartn_z'])
        iocc = cidx(['_atom_site.occupancy'])
        ib = cidx(['_atom_site.b_iso_or_equiv'])
        ielem = cidx(['_atom_site.type_symbol'])

        # Tokenise data rows (handle quoted strings)
        tokens = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", data_block)
        n_cols = len(cols)
        if n_cols == 0:
            return atoms

        serial = 0
        for i in range(0, len(tokens) - n_cols + 1, n_cols):
            row = tokens[i:i + n_cols]
            if len(row) < n_cols:
                break

            def get(idx, default=''):
                if idx is None or idx >= len(row):
                    return default
                v = row[idx].strip("'\"")
                return '' if v in ('.', '?') else v

            record = get(igroup, 'ATOM').upper()
            if record not in ('ATOM', 'HETATM'):
                continue
            serial += 1
            try:
                b_str = get(ib, '0')
                b_iso = float(b_str) if b_str else 0.0
                occ_str = get(iocc, '1')
                occupancy = float(occ_str) if occ_str else 1.0
                seq_str = get(iseq, '0')
                res_seq = int(seq_str) if seq_str.lstrip('-').isdigit() else 0
                element = get(ielem, '').upper()
                if not element:
                    element = re.sub(r'[^A-Za-z]', '', get(iname, 'N'))[:2].upper()

                atom = Atom(
                    serial=serial,
                    name=get(iname),
                    alt_loc=get(ialt),
                    res_name=get(iresname),
                    chain_id=get(ichain),
                    res_seq=res_seq,
                    ins_code=get(iins),
                    x=float(get(ixx, '0')),
                    y=float(get(iyy, '0')),
                    z=float(get(izz, '0')),
                    occupancy=occupancy,
                    b_iso=b_iso,
                    element=element,
                    record_type=record,
                )
                atoms.append(atom)
            except (ValueError, IndexError):
                continue

        return atoms


# ---------------------------------------------------------------------------
# Gemmi-based parser (preferred when gemmi is installed)
# ---------------------------------------------------------------------------

try:
    import gemmi as _gemmi
    _GEMMI_AVAILABLE = True
    _GEMMI_ELEMENT_X = _gemmi.Element('X')  # Sentinel for unknown/unrecognised element
except ImportError:
    _gemmi = None  # type: ignore
    _GEMMI_AVAILABLE = False
    _GEMMI_ELEMENT_X = None  # type: ignore


class GemmiParser:
    """Parse PDB/mmCIF files using the gemmi library.

    Uses gemmi for atom extraction (both formats) and for refinement params
    from mmCIF.  For PDB-format files, refinement params fall back to the
    existing PDBParser._parse_remark3() regex logic since gemmi's PDB
    REMARK 3 parsing is less complete than its mmCIF support.

    Raises RuntimeError if gemmi is not installed.
    """

    @classmethod
    def parse(cls, filepath: str) -> Tuple[List[Atom], RefinementParams]:
        if not _GEMMI_AVAILABLE:
            raise RuntimeError(
                "gemmi is not installed.  Install it with: pip install gemmi"
            )
        st = _gemmi.read_structure(filepath)
        atoms = cls._extract_atoms(st)

        is_cif = filepath.lower().endswith(('.cif', '.mmcif'))
        if is_cif:
            doc = _gemmi.cif.read(filepath)
            params = cls._extract_params_cif(st, doc)
        else:
            # PDB format: use existing regex parser for REMARK 3 stats
            params = RefinementParams(source='PDB REMARK 3 (via GemmiParser)')
            remark3_lines = []
            with open(filepath, 'r', errors='replace') as fh:
                for line in fh:
                    rec = line[:6].strip()
                    if rec == 'REMARK' and line[6:10].strip() == '3':
                        remark3_lines.append(line[10:].rstrip())
            PDBParser._parse_remark3(remark3_lines, params)

        return atoms, params

    @classmethod
    def _extract_atoms(cls, st) -> List[Atom]:
        atoms = []
        model = st[0]
        serial = 0
        for chain in model:
            for residue in chain:
                for ga in residue:
                    serial += 1
                    alt = ga.altloc if ga.altloc != '\x00' else ''
                    element = ga.element.name if ga.element != _GEMMI_ELEMENT_X else ''
                    if not element or element == 'X':
                        # Fall back to inferring element from atom name; default to 'C' (carbon)
                        # as it is the most common element in macromolecular structures.
                        element = re.sub(r'[^A-Za-z]', '', ga.name)[:2].upper() or 'C'
                    record = 'HETATM' if residue.het_flag == 'H' else 'ATOM'
                    ins = residue.seqid.icode.strip() if residue.seqid.icode.strip() else ''
                    try:
                        seq_num = int(str(residue.seqid.num))
                    except (ValueError, AttributeError):
                        seq_num = 0
                    atoms.append(Atom(
                        serial=serial,
                        name=ga.name,
                        alt_loc=alt,
                        res_name=residue.name,
                        chain_id=chain.name,
                        res_seq=seq_num,
                        ins_code=ins,
                        x=ga.pos.x,
                        y=ga.pos.y,
                        z=ga.pos.z,
                        occupancy=ga.occ,
                        b_iso=ga.b_iso,
                        element=element.upper(),
                        record_type=record,
                    ))
        return atoms

    @classmethod
    def _extract_params_cif(cls, st, doc) -> RefinementParams:
        params = RefinementParams(source='mmCIF (gemmi)')

        # Try structure-level resolution first
        if st.resolution > 0:
            params.resolution = st.resolution

        block = doc.sole_block()

        def fval(key):
            v = block.find_value(key)
            if v and v not in ('.', '?', ''):
                try:
                    return float(v)
                except ValueError:
                    pass
            return None

        def ival(key):
            v = fval(key)
            return int(v) if v is not None else None

        def fval_any(keys):
            for k in keys:
                v = fval(k)
                if v is not None:
                    return v
            return None

        # Resolution
        res = fval_any(['_refine.ls_d_res_high', '_reflns.d_resolution_high'])
        if res is not None:
            params.resolution = res

        # R_work
        params.r_work = fval_any([
            '_refine.ls_r_factor_r_work',
            '_refine.ls_r_factor_obs',
            '_refine.ls_r_factor_all',
        ])

        # R_free
        params.r_free = fval('_refine.ls_r_factor_r_free')

        # Completeness
        comp = fval_any([
            '_reflns.percent_possible_all',
            '_reflns.pdbx_percent_possible_all',
            '_reflns.pdbx_percent_possible_obs',
        ])
        if comp is not None:
            params.completeness = comp / 100.0 if comp > 1 else comp

        # N_obs
        params.n_obs = ival('_refine.ls_number_reflns_obs') or ival('_refine.ls_number_reflns_all')

        # N_free
        params.n_free = ival('_refine.ls_number_reflns_r_free')

        # N_params
        params.n_params = ival('_refine.ls_number_parameters')

        # N_atoms
        n_atoms = ival('_refine_hist.number_atoms_total') or ival('_refine.ls_number_atoms_total')
        if n_atoms is not None:
            params.n_atoms_refined = n_atoms

        # Space group
        sg = block.find_value('_symmetry.space_group_name_h-m') or \
             block.find_value('_space_group.name_h-m_alt') or ''
        params.space_group = sg.strip("'\" ") if sg not in ('.', '?') else ''

        return params


# ---------------------------------------------------------------------------
# Phenix log parser
# ---------------------------------------------------------------------------

class PhenixLogParser:
    """Parse phenix.refine output log files for refinement statistics.

    Phenix logs contain refinement statistics in a tabular format that differs
    from PDB REMARK 3.  This parser extracts R_work, R_free, N_obs, N_params,
    completeness, and resolution from the final refinement cycle reported.
    """

    @classmethod
    def parse(cls, filepath: str) -> RefinementParams:
        params = RefinementParams(source='Phenix log')
        with open(filepath, 'r', errors='replace') as fh:
            content = fh.read()
        cls._extract(content, params)
        return params

    @classmethod
    def _extract(cls, content: str, params: RefinementParams):
        # Resolution
        for pat in [
            r'd_min\s*=\s*([\d.]+)',
            r'High resolution limit\s*:\s*([\d.]+)',
            r'resolution\s*\(high\)\s*=\s*([\d.]+)',
            r'high_resolution\s*=\s*([\d.]+)',
        ]:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                params.resolution = float(m.group(1))
                break

        # R_work — take the last occurrence (final cycle)
        r_work_vals = re.findall(
            r'r_work\s*=\s*([\d.]+)', content, re.IGNORECASE)
        if not r_work_vals:
            r_work_vals = re.findall(
                r'R-work\s*=\s*([\d.]+)', content, re.IGNORECASE)
        if r_work_vals:
            v = float(r_work_vals[-1])
            params.r_work = v if v < 1 else v / 100.0

        # R_free — take the last occurrence (final cycle)
        r_free_vals = re.findall(
            r'r_free\s*=\s*([\d.]+)', content, re.IGNORECASE)
        if not r_free_vals:
            r_free_vals = re.findall(
                r'R-free\s*=\s*([\d.]+)', content, re.IGNORECASE)
        if r_free_vals:
            v = float(r_free_vals[-1])
            params.r_free = v if v < 1 else v / 100.0

        # Number of reflections used in refinement
        for pat in [
            r'number_of_reflections\s*=\s*(\d+)',
            r'Number of reflections used in refinement\s*:\s*(\d+)',
            r'Reflections used in refinement\s*:\s*(\d+)',
            r'n_obs\s*=\s*(\d+)',
        ]:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                params.n_obs = int(m.group(1))
                break

        # Number of free reflections
        for pat in [
            r'number_of_test_reflections\s*=\s*(\d+)',
            r'Number of test set reflections\s*:\s*(\d+)',
            r'n_free\s*=\s*(\d+)',
        ]:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                params.n_free = int(m.group(1))
                break

        # Number of parameters
        for pat in [
            r'Number of parameters\s*:\s*(\d+)',
            r'number_of_parameters\s*=\s*(\d+)',
        ]:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                params.n_params = int(m.group(1))
                break

        # Completeness
        for pat in [
            r'completeness\s*\(%\)\s*=\s*([\d.]+)',
            r'Completeness\s*\(%\)\s*:\s*([\d.]+)',
            r'completeness_in_range\s*=\s*([\d.]+)',
        ]:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                params.completeness = v / 100.0 if v > 1 else v
                break

        # Number of atoms
        for pat in [
            r'Number of atoms\s*:\s*(\d+)',
            r'n_atoms\s*=\s*(\d+)',
        ]:
            m = re.search(pat, content, re.IGNORECASE)
            if m:
                params.n_atoms_refined = int(m.group(1))
                break


# ---------------------------------------------------------------------------
# RCSB downloader
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, dest_dir: str = '.', prefer_mmcif: bool = True) -> str:
    """Download a structure from RCSB PDB."""
    pdb_id = pdb_id.strip().upper()
    if prefer_mmcif:
        url = f'https://files.rcsb.org/download/{pdb_id}.cif'
        dest = os.path.join(dest_dir, f'{pdb_id}.cif')
    else:
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        dest = os.path.join(dest_dir, f'{pdb_id}.pdb')
    print(f"Downloading {pdb_id} from {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
        return dest
    except Exception as e:
        raise RuntimeError(f"Failed to download {pdb_id}: {e}")


# ---------------------------------------------------------------------------
# DPI Calculator
# ---------------------------------------------------------------------------

class DPICalculator:
    """
    Computes Cruickshank DPI (overall) and per-atom coordinate precision.

    Two overall DPI formulae are implemented:
        R-based:     σ(x) = sqrt(N_a / p)     × C^(-1/3) × R      × d_min
        Rfree-based: σ(x) = sqrt(N_a / n_obs) × C^(-1/3) × R_free × d_min

    The R_free formula uses n_obs (total observed reflections) as the denominator,
    per Cruickshank (1999) Eq. 31 and Table 3 — NOT N_free (free-set size).

    Per-atom (Helliwell 2023, Eq. 2):
        σ(x, B_i) = σ(x, B_avg) × (Z_avg / Z_i) × √(B_i / B_avg)
    """

    def __init__(
        self,
        atoms: List[Atom],
        params: RefinementParams,
        include_hetatm: bool = False,
        include_hydrogens: bool = False,
        min_occupancy: float = 0.5,
        apply_z_correction: bool = True,
        scale_factor: float = 1.0,  # 0.65 or 1.0 (Cruickshank recommends 1.0 for caution)
    ):
        self.atoms = atoms
        self.params = params
        self.include_hetatm = include_hetatm
        self.include_hydrogens = include_hydrogens
        self.min_occupancy = min_occupancy
        self.apply_z_correction = apply_z_correction
        self.scale_factor = scale_factor

    def _working_atoms(self) -> List[Atom]:
        """Atoms included in the DPI calculation.

        Alternate conformers B, C, … are excluded; only the primary conformer
        (alt_loc == '' or 'A') is kept to avoid inflating N_a and skewing B_avg.
        """
        result = []
        for a in self.atoms:
            if a.is_hydrogen and not self.include_hydrogens:
                continue
            if a.record_type == 'HETATM' and not self.include_hetatm:
                continue
            if a.occupancy < self.min_occupancy:
                continue
            # Exclude non-primary alternate conformers (B, C, …)
            if a.alt_loc not in ('', 'A'):
                continue
            result.append(a)
        return result

    def _b_average(self, working_atoms: List[Atom]) -> float:
        if not working_atoms:
            return 20.0
        return sum(a.b_iso for a in working_atoms) / len(working_atoms)

    def _z_average(self, working_atoms: List[Atom]) -> float:
        """Scattering-weighted mean Z (simple mean for typical proteins)."""
        zs = [a.atomic_number for a in working_atoms if not a.is_hydrogen]
        if not zs:
            return 7.0
        return sum(zs) / len(zs)

    def _n_atoms(self, working_atoms: List[Atom]) -> int:
        return self.params.n_atoms_refined or len(working_atoms)

    def _check_params(self) -> None:
        """Emit warnings for suspicious refinement parameter values."""
        p = self.params
        if p.r_work is not None and p.r_free is not None:
            if p.r_work > p.r_free:
                print(f"  Warning: R_work ({p.r_work:.4f}) > R_free ({p.r_free:.4f}) — "
                      f"this is unusual and may indicate a parsing error.")
        if p.r_work is not None:
            if p.r_work > 0.6:
                print(f"  Warning: R_work ({p.r_work:.4f}) > 0.6 — likely a parsing error "
                      f"or structure is not properly refined.")
            elif p.r_work < 0.01:
                print(f"  Warning: R_work ({p.r_work:.4f}) < 0.01 — likely a parsing error.")
        if p.completeness is not None and p.completeness < 0.5:
            print(f"  Warning: Data completeness ({p.completeness:.1%}) < 50% — "
                  f"DPI values may be unreliable.")
        if p.resolution is None:
            print("  Warning: Resolution (d_min) is missing; DPI cannot be computed.")
        elif p.resolution <= 0 or p.resolution > 10:
            print(f"  Warning: Resolution ({p.resolution} Å) appears unreasonable.")

    def _estimate_n_params(self, n_atoms: int) -> Optional[int]:
        """Auto-estimate N_params when not reported.

        Standard approximation:
            isotropic B:   N_params ≈ 4 × N_atoms  (x, y, z, B per atom)
            anisotropic B: N_params ≈ 10 × N_atoms (x, y, z, 6×Uij, occ per atom)
        """
        estimated = 4 * n_atoms
        print(f"  Note: N_params not found in file; auto-estimating as "
              f"4 × N_atoms = {estimated} (isotropic B assumption). "
              f"Override with --n-params if needed.")
        return estimated

    def calculate_r_based(self) -> Optional[DPIResult]:
        """Compute DPI using R_work and p = N_obs − N_params."""
        p = self.params
        self._check_params()
        if p.r_work is None or p.resolution is None:
            return None
        n_obs = p.n_obs
        if n_obs is None:
            return None
        working = self._working_atoms()
        n_a = self._n_atoms(working)
        n_params = p.n_params
        if n_params is None:
            n_params = self._estimate_n_params(n_a)
        dof = n_obs - n_params
        if dof <= 0:
            print("  Warning: p = N_obs − N_params ≤ 0; R-based DPI is not valid. Use R_free formula.")
            return None
        comp = p.completeness or 1.0
        sigma_x = self.scale_factor * math.sqrt(n_a / dof) * (comp ** (-1/3)) * p.r_work * p.resolution
        sigma_r = math.sqrt(3) * sigma_x
        b_avg = self._b_average(working)
        z_avg = self._z_average(working)
        per_atom = self._per_atom(working, sigma_x, b_avg, z_avg, p.resolution)
        return DPIResult(
            params=p, method='R', sigma_x_avg=sigma_x, sigma_r_avg=sigma_r,
            p=dof, b_avg=b_avg, z_avg=z_avg,
            n_atoms_used=len(working), per_atom=per_atom,
        )

    def calculate_rfree_based(self) -> Optional[DPIResult]:
        """Compute DPI using R_free and n_obs (total observed reflections).

        Per Cruickshank (1999) Eq. 31 and Table 3, the R_free formula uses n_obs
        (total observed reflections) as the denominator — NOT N_free (the free-set
        size).  Table 3 lists (Nᵢ/n_obs)^½ explicitly in the R_free rows, confirming
        this choice.
        """
        p = self.params
        if p.r_free is None or p.resolution is None:
            return None
        working = self._working_atoms()
        n_a_val = p.n_atoms_refined
        if n_a_val is None:
            n_a_val = len(working)

        n_obs = p.n_obs
        if n_obs is None:
            return None

        comp = p.completeness or 1.0
        sigma_x = self.scale_factor * math.sqrt(n_a_val / n_obs) * (comp ** (-1/3)) * p.r_free * p.resolution
        sigma_r = math.sqrt(3) * sigma_x
        b_avg = self._b_average(working)
        z_avg = self._z_average(working)
        per_atom = self._per_atom(working, sigma_x, b_avg, z_avg, p.resolution)
        return DPIResult(
            params=p, method='Rfree', sigma_x_avg=sigma_x, sigma_r_avg=sigma_r,
            p=None, b_avg=b_avg, z_avg=z_avg,
            n_atoms_used=len(working), per_atom=per_atom,
        )

    def _per_atom(
        self, working: List[Atom], sigma_x_avg: float, b_avg: float,
        z_avg: float, d_min: float
    ) -> List[AtomDPI]:
        result = []
        for atom in working:
            # B-factor correction (Helliwell 2023, Eq. 2): σ(x, Bᵢ) = σ(x, B_avg) × √(Bᵢ / B_avg)
            if b_avg > 0 and atom.b_iso > 0:
                b_corr = math.sqrt(atom.b_iso / b_avg)
            else:
                b_corr = 1.0
            # Atomic number correction (optional, from Gurusaran et al. 2014)
            if self.apply_z_correction:
                z_corr = z_avg / max(atom.atomic_number, 1)
            else:
                z_corr = 1.0
            sigma_x_i = sigma_x_avg * z_corr * b_corr
            sigma_r_i = math.sqrt(3) * sigma_x_i
            result.append(AtomDPI(atom=atom, sigma_x=sigma_x_i, sigma_r=sigma_r_i))
        return result

    def calculate_all(self) -> Dict[str, Optional[DPIResult]]:
        return {
            'R': self.calculate_r_based(),
            'Rfree': self.calculate_rfree_based(),
        }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def print_summary(result: DPIResult, title: str = ''):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
    print(f"  Method: {result.method}-based DPI")
    print(f"{'='*60}")
    print(f"  Overall σ(x, B_avg): {result.sigma_x_avg:.4f} Å")
    print(f"  Overall σ(r, B_avg): {result.sigma_r_avg:.4f} Å  [= √3 × σ(x)]")
    print(f"  B_avg (working atoms): {result.b_avg:.2f} Å²")
    print(f"  Z_avg (mean atomic Z): {result.z_avg:.2f}")
    print(f"  Atoms used:            {result.n_atoms_used}")
    if result.p is not None:
        print(f"  p = N_obs − N_params:  {result.p}")


def write_csv(result: DPIResult, filepath: str):
    """Write per-atom DPI values to CSV."""
    with open(filepath, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'serial', 'name', 'alt_loc', 'res_name', 'chain', 'res_seq',
            'ins_code', 'element', 'occupancy', 'B_iso', 'sigma_x_A', 'sigma_r_A',
        ])
        for ad in result.per_atom:
            a = ad.atom
            writer.writerow([
                a.serial, a.name, a.alt_loc, a.res_name, a.chain_id, a.res_seq,
                a.ins_code, a.element, f'{a.occupancy:.2f}', f'{a.b_iso:.2f}',
                f'{ad.sigma_x:.4f}', f'{ad.sigma_r:.4f}',
            ])
    print(f"  Per-atom CSV written: {filepath}")


def write_annotated_pdb(
    result: DPIResult, input_filepath: str, output_filepath: str,
    replace_b_with: str = 'sigma_x',  # or 'sigma_r'
):
    """
    Write an annotated PDB file where the B-factor column is replaced
    with σ(x) or σ(r) for each atom, and a REMARK is added.
    """
    # Build lookup: serial → AtomDPI
    lookup = {ad.atom.serial: ad for ad in result.per_atom}

    lines_out = []
    col_label = 'sigma(x)' if replace_b_with == 'sigma_x' else 'sigma(r)'

    # Header remarks
    lines_out.append(
        f"REMARK   1 DPI CALCULATION – {result.method}-based Cruickshank DPI\n"
    )
    lines_out.append(
        f"REMARK   1 Overall sigma(x,B_avg) = {result.sigma_x_avg:.4f} A\n"
    )
    lines_out.append(
        f"REMARK   1 Overall sigma(r,B_avg) = {result.sigma_r_avg:.4f} A\n"
    )
    lines_out.append(
        f"REMARK   1 B-factor column replaced with per-atom {col_label} (Angstroms)\n"
    )
    lines_out.append(
        "REMARK   1 Ref: Cruickshank (1999) D55; Gurusaran et al. (2014) IUCrJ 1\n"
    )

    with open(input_filepath, 'r', errors='replace') as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ('ATOM', 'HETATM'):
                try:
                    serial = int(line[6:11].strip())
                    if serial in lookup:
                        ad = lookup[serial]
                        val = ad.sigma_x if replace_b_with == 'sigma_x' else ad.sigma_r
                        line = line[:60] + f'{val:6.2f}' + line[66:]
                except (ValueError, IndexError):
                    pass
            lines_out.append(line if line.endswith('\n') else line + '\n')

    with open(output_filepath, 'w') as fh:
        fh.writelines(lines_out)
    print(f"  Annotated PDB written: {output_filepath}")


def print_stats_table(result: DPIResult, n_top: int = 20):
    """Print top/bottom atoms by precision."""
    sorted_atoms = sorted(result.per_atom, key=lambda x: x.sigma_x)
    print(f"\n{'─'*75}")
    print(f"  Top {n_top} most precisely determined atoms (lowest σ(x)):")
    print(f"  {'Serial':>7} {'Chain':>5} {'Res':>5} {'ResName':>7} {'Atom':>5} {'Elem':>4}  "
          f"{'B_iso':>7}  {'σ(x)':>8}  {'σ(r)':>8}")
    print(f"  {'─'*73}")
    for ad in sorted_atoms[:n_top]:
        a = ad.atom
        print(f"  {a.serial:>7} {a.chain_id:>5} {a.res_seq:>5} {a.res_name:>7} {a.name:>5} "
              f"{a.element:>4}  {a.b_iso:>7.2f}  {ad.sigma_x:>8.4f}  {ad.sigma_r:>8.4f}")

    print(f"\n  Bottom {n_top} least precisely determined atoms (highest σ(x)):")
    print(f"  {'Serial':>7} {'Chain':>5} {'Res':>5} {'ResName':>7} {'Atom':>5} {'Elem':>4}  "
          f"{'B_iso':>7}  {'σ(x)':>8}  {'σ(r)':>8}")
    print(f"  {'─'*73}")
    for ad in sorted_atoms[-n_top:]:
        a = ad.atom
        print(f"  {a.serial:>7} {a.chain_id:>5} {a.res_seq:>5} {a.res_name:>7} {a.name:>5} "
              f"{a.element:>4}  {a.b_iso:>7.2f}  {ad.sigma_x:>8.4f}  {ad.sigma_r:>8.4f}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--pdb-id', metavar='XXXX',
                     help='4-letter PDB accession code (downloads from RCSB)')
    src.add_argument('--file', metavar='PATH',
                     help='Path to PDB or mmCIF file')

    p.add_argument('--phenix-log', metavar='PATH',
                   help='Path to a phenix.refine output log file to extract '
                        'refinement statistics (overrides parsed REMARK 3 values)')
    p.add_argument('--out-dir', metavar='DIR', default='.',
                   help='Output directory (default: current directory)')
    p.add_argument('--prefix', metavar='STR', default='',
                   help='Output filename prefix')

    # Manual parameter overrides
    ov = p.add_argument_group('Manual parameter overrides (override file-parsed values)')
    ov.add_argument('--resolution', type=float, help='Resolution d_min (Å)')
    ov.add_argument('--r-work', type=float, help='R_work (0–1 scale)')
    ov.add_argument('--r-free', type=float, help='R_free (0–1 scale)')
    ov.add_argument('--completeness', type=float, help='Data completeness (0–1 or 0–100)')
    ov.add_argument('--n-obs', type=int, help='Number of observed reflections in refinement')
    ov.add_argument('--n-params', type=int, help='Number of refined parameters')
    ov.add_argument('--n-free', type=int, help='Number of free reflections')
    ov.add_argument('--n-atoms', type=int, help='Number of atoms in refinement')

    p.add_argument('--scale-factor', type=float, default=1.0,
                   help='Prefactor (0.65 or 1.0; default 1.0 per Cruickshank recommendation)')
    p.add_argument('--no-z-correction', action='store_true',
                   help='Omit the (Z_avg/Z_i) atomic number correction in per-atom DPI')
    p.add_argument('--include-hetatm', action='store_true',
                   help='Include HETATM records in per-atom calculation')
    p.add_argument('--include-hydrogens', action='store_true',
                   help='Include hydrogen atoms')
    p.add_argument('--min-occupancy', type=float, default=0.5,
                   help='Minimum occupancy for atoms to include (default 0.5)')
    p.add_argument('--annotate-sigma', choices=['sigma_x', 'sigma_r'], default='sigma_x',
                   help='Which value to put in annotated PDB B-column (default: sigma_x)')
    p.add_argument('--top', type=int, default=10,
                   help='Number of top/bottom atoms to display (default 10)')
    p.add_argument('--method', choices=['R', 'Rfree', 'both'], default='both',
                   help='Which DPI formula to use (default: both)')
    p.add_argument('--download-pdb', action='store_true',
                   help='Prefer PDB format when downloading (default: mmCIF)')
    return p


def apply_overrides(params: RefinementParams, args) -> RefinementParams:
    # Merge in Phenix log statistics (lower priority than explicit CLI overrides)
    if getattr(args, 'phenix_log', None):
        phenix_params = PhenixLogParser.parse(args.phenix_log)
        print(f"\nPhenix log parsed: {args.phenix_log}")
        # Only overwrite fields that are currently missing
        if params.resolution is None and phenix_params.resolution is not None:
            params.resolution = phenix_params.resolution
        if params.r_work is None and phenix_params.r_work is not None:
            params.r_work = phenix_params.r_work
        if params.r_free is None and phenix_params.r_free is not None:
            params.r_free = phenix_params.r_free
        if params.completeness is None and phenix_params.completeness is not None:
            params.completeness = phenix_params.completeness
        if params.n_obs is None and phenix_params.n_obs is not None:
            params.n_obs = phenix_params.n_obs
        if params.n_free is None and phenix_params.n_free is not None:
            params.n_free = phenix_params.n_free
        if params.n_params is None and phenix_params.n_params is not None:
            params.n_params = phenix_params.n_params
        if params.n_atoms_refined is None and phenix_params.n_atoms_refined is not None:
            params.n_atoms_refined = phenix_params.n_atoms_refined

    if args.resolution is not None:
        params.resolution = args.resolution
    if args.r_work is not None:
        params.r_work = args.r_work
    if args.r_free is not None:
        params.r_free = args.r_free
    if args.completeness is not None:
        v = args.completeness
        params.completeness = v / 100.0 if v > 1 else v
    if args.n_obs is not None:
        params.n_obs = args.n_obs
    if args.n_params is not None:
        params.n_params = args.n_params
    if args.n_free is not None:
        params.n_free = args.n_free
    if args.n_atoms is not None:
        params.n_atoms_refined = args.n_atoms
    return params


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve input file
    if args.pdb_id:
        pdb_id = args.pdb_id.upper()
        prefix = args.prefix or pdb_id
        prefer_pdb = args.download_pdb
        filepath = download_pdb(pdb_id, dest_dir=args.out_dir, prefer_mmcif=not prefer_pdb)
    else:
        filepath = args.file
        prefix = args.prefix or Path(filepath).stem

    # Parse
    ext = Path(filepath).suffix.lower()
    print(f"\nParsing: {filepath}")
    if _GEMMI_AVAILABLE:
        atoms, params = GemmiParser.parse(filepath)
    elif ext in ('.cif', '.mmcif'):
        atoms, params = MMCIFParser.parse(filepath)
    else:
        atoms, params = PDBParser.parse(filepath)

    # Apply overrides
    params = apply_overrides(params, args)
    print(params.summary())

    if not atoms:
        print("ERROR: No atoms parsed from file.", file=sys.stderr)
        sys.exit(1)
    print(f"\nAtoms parsed: {len(atoms)} total")

    # Calculate
    calc = DPICalculator(
        atoms=atoms,
        params=params,
        include_hetatm=args.include_hetatm,
        include_hydrogens=args.include_hydrogens,
        min_occupancy=args.min_occupancy,
        apply_z_correction=not args.no_z_correction,
        scale_factor=args.scale_factor,
    )

    results = {}
    if args.method in ('R', 'both'):
        r = calc.calculate_r_based()
        if r:
            results['R'] = r
        else:
            print("  R-based DPI: not available (missing parameters or p ≤ 0)")
    if args.method in ('Rfree', 'both'):
        r = calc.calculate_rfree_based()
        if r:
            results['Rfree'] = r
        else:
            print("  R_free-based DPI: not available (missing R_free or N_free)")

    if not results:
        print("\nERROR: Could not compute DPI. Check that R/R_free, resolution,\n"
              "completeness, and reflection counts are available in the file\n"
              "or supplied via command-line overrides.", file=sys.stderr)
        sys.exit(1)

    # Print summaries and write outputs
    for method, result in results.items():
        print_summary(result, title=f"{prefix} – {method}-based DPI")
        print_stats_table(result, n_top=args.top)

        csv_path = os.path.join(args.out_dir, f'{prefix}_dpi_{method.lower()}.csv')
        write_csv(result, csv_path)

        # Annotated PDB only makes sense for PDB input files
        if ext not in ('.cif', '.mmcif'):
            ann_path = os.path.join(
                args.out_dir, f'{prefix}_dpi_{method.lower()}_annotated.pdb')
            write_annotated_pdb(result, filepath, ann_path, args.annotate_sigma)

    print(f"\nDone. Output files in: {os.path.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()
