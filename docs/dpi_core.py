"""
dpi_core.py — Self-contained DPI calculation logic for Pyodide (no gemmi).

This module is a stripped-down version of dpi_calculator.py that can run
entirely in the browser via Pyodide (CPython compiled to WebAssembly).

Implements Cruickshank (1999) DPI with per-atom extension per Helliwell (2023).

References
----------
Cruickshank, D. W. J. (1999). Acta Cryst. D55, 583-601.
Gurusaran, M. et al. (2014). IUCrJ 1, 74-81.
Helliwell, J. R. (2023). Curr. Res. Struct. Biol. 6, 100111.
"""

import math
import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ---------------------------------------------------------------------------
# Atomic numbers
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
    record_type: str  # 'ATOM' or 'HETATM'

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
    resolution: Optional[float] = None
    r_work: Optional[float] = None
    r_free: Optional[float] = None
    completeness: Optional[float] = None
    n_obs: Optional[int] = None
    n_params: Optional[int] = None
    n_free: Optional[int] = None
    n_atoms_refined: Optional[int] = None
    space_group: str = ''
    source: str = ''


@dataclass
class AtomDPI:
    atom: Atom
    sigma_x: float
    sigma_r: float


@dataclass
class DPIResult:
    params: RefinementParams
    method: str
    sigma_x_avg: float
    sigma_r_avg: float
    p: Optional[int]
    b_avg: float
    z_avg: float
    n_atoms_used: int
    per_atom: List[AtomDPI] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDB parser
# ---------------------------------------------------------------------------

class PDBParser:
    """Parse PDB flat files for atoms and REMARK 3 refinement statistics."""

    _REMARK3_PATTERNS = {
        'resolution': [
            r'RESOLUTION RANGE HIGH \(ANGSTROMS\)\s*:\s*([\d.]+)',
            r'RESOLUTION\s*[:(]\s*([\d.]+)',
        ],
        'r_work': [
            r'R VALUE\s+\(WORKING SET\)\s*:\s*([\d.]+)',
            r'R VALUE\s+\(WORKING\)\s*:\s*([\d.]+)',
            r'R VALUE\s+\(WORKING SET,\s*NO CUTOFF\)\s*:\s*([\d.]+)',
        ],
        'r_free': [
            r'FREE R VALUE\s*:\s*([\d.]+)',
            r'R FREE\s*:\s*([\d.]+)',
        ],
        'completeness': [
            r'COMPLETENESS FOR RANGE\s*\(%\)\s*:\s*([\d.]+)',
            r'COMPLETENESS\s*\(%\)\s*:\s*([\d.]+)',
        ],
        'n_obs': [
            r'NUMBER OF REFLECTIONS\s*:\s*(\d+)',
            r'REFLECTIONS USED IN REFINEMENT\s*:\s*(\d+)',
        ],
        'n_params': [
            r'NUMBER OF PARAMETERS IN REFINEMENT\s*:\s*(\d+)',
        ],
    }

    @classmethod
    def parse(cls, content: str) -> Tuple[List[Atom], RefinementParams]:
        atoms = []
        params = RefinementParams(source='PDB REMARK 3')
        remark3_lines = []

        for line in content.splitlines():
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
            element = line[76:78].strip() if len(line) > 76 else ''
            if not element:
                raw = re.sub(r'^\d+', '', name.strip())
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

        n_free = None
        for pat in [
            r'FREE R VALUE TEST SET COUNT\s*:\s*(\d+)',
            r'NUMBER OF FREE REFLECTIONS\s*:\s*(\d+)',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                n_free = int(m.group(1))
                break
        if n_free:
            params.n_free = n_free
        else:
            pct_m = re.search(
                r'FREE R VALUE TEST SET SIZE\s*\(%\)\s*:\s*([\d.]+)', text, re.IGNORECASE)
            if pct_m and params.n_obs:
                params.n_free = int(round(float(pct_m.group(1)) / 100.0 * params.n_obs))


# ---------------------------------------------------------------------------
# mmCIF parser
# ---------------------------------------------------------------------------

class MMCIFParser:
    """Parse mmCIF files for atoms and refinement statistics."""

    @classmethod
    def parse(cls, content: str) -> Tuple[List[Atom], RefinementParams]:
        params = RefinementParams(source='mmCIF _refine/_reflns')

        scalar_map = cls._extract_scalars(content)
        cls._fill_params(scalar_map, params)
        atoms = cls._extract_atoms(content)

        return atoms, params

    @classmethod
    def _extract_scalars(cls, content: str) -> Dict[str, str]:
        result = {}
        # Match data items that start at a word boundary (beginning of line or after whitespace)
        # to avoid false matches inside tokens like 'data_ENTRY' matching '_ENTRY'.
        pattern = re.compile(
            r'(?:^|\s)(_[\w.]+)\s+([\'"])(.*?)\2|(?:^|\s)(_[\w.]+)\s+(\S+)',
            re.DOTALL | re.MULTILINE
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
        ])
        params.r_work = fget([
            '_refine.ls_r_factor_r_work',
            '_refine.ls_r_factor_obs',
            '_refine.ls_r_factor_all',
        ])
        params.r_free = fget(['_refine.ls_r_factor_r_free'])
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
        params.n_free = iget(['_refine.ls_number_reflns_r_free'])
        params.n_params = iget(['_refine.ls_number_parameters'])
        params.n_atoms_refined = iget([
            '_refine.ls_number_atoms_total',
            '_refine_hist.number_atoms_total',
        ])

        sg = d.get('_symmetry.space_group_name_h-m') or d.get('_space_group.name_h-m_alt', '')
        params.space_group = sg.strip("'\" ")

    @classmethod
    def _extract_atoms(cls, content: str) -> List[Atom]:
        atoms = []
        # Find the atom_site loop block using a simple string search to avoid ReDoS.
        # Locate "loop_" followed by _atom_site. columns, then the data rows.
        loop_start = -1
        search_from = 0
        lower_content = content.lower()
        while True:
            idx = lower_content.find('loop_', search_from)
            if idx == -1:
                break
            # Check if the next non-whitespace token is an _atom_site. column
            rest = content[idx + 5:].lstrip()
            if rest.lower().startswith('_atom_site.'):
                loop_start = idx
                break
            search_from = idx + 5
        if loop_start == -1:
            return atoms

        # Extract header columns and data block manually
        pos = loop_start + 5  # skip 'loop_'
        header_cols = []
        # Collect _atom_site.xxx tokens from the header
        while True:
            chunk = content[pos:].lstrip()
            m = re.match(r'(_atom_site\.\S+)', chunk, re.IGNORECASE)
            if m:
                header_cols.append(m.group(1).lower())
                pos += len(content[pos:]) - len(chunk) + m.end()
            else:
                break
        if not header_cols:
            return atoms

        # The data block starts at pos; it ends at the next 'loop_' or end-of-string
        next_loop = lower_content.find('loop_', pos)
        data_block = content[pos:next_loop].strip() if next_loop != -1 else content[pos:].strip()

        col_idx = {c: i for i, c in enumerate(header_cols)}

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

        tokens = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", data_block)
        n_cols = len(header_cols)
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
# DPI Calculator
# ---------------------------------------------------------------------------

class DPICalculator:
    """
    Computes Cruickshank DPI (overall) and per-atom coordinate precision.

    R-based:     σ(x) = sqrt(N_a / p)     × C^(-1/3) × R      × d_min
    Rfree-based: σ(x) = sqrt(N_a / n_obs) × C^(-1/3) × R_free × d_min

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
    ):
        self.atoms = atoms
        self.params = params
        self.include_hetatm = include_hetatm
        self.include_hydrogens = include_hydrogens
        self.min_occupancy = min_occupancy
        self.apply_z_correction = apply_z_correction

    def _working_atoms(self) -> List[Atom]:
        result = []
        for a in self.atoms:
            if a.is_hydrogen and not self.include_hydrogens:
                continue
            if a.record_type == 'HETATM' and not self.include_hetatm:
                continue
            if a.occupancy < self.min_occupancy:
                continue
            if a.alt_loc not in ('', 'A'):
                continue
            result.append(a)
        return result

    def _output_atoms(self) -> List[Atom]:
        result = []
        for a in self.atoms:
            if a.is_hydrogen and not self.include_hydrogens:
                continue
            if a.record_type == 'HETATM' and not self.include_hetatm:
                continue
            result.append(a)
        return result

    def _b_average(self, working_atoms: List[Atom]) -> float:
        if not working_atoms:
            return 20.0
        return sum(a.b_iso for a in working_atoms) / len(working_atoms)

    def _z_average(self, working_atoms: List[Atom]) -> float:
        zs = [a.atomic_number for a in working_atoms if not a.is_hydrogen]
        if not zs:
            return 7.0
        return sum(zs) / len(zs)

    def _n_atoms(self, working_atoms: List[Atom]) -> int:
        return self.params.n_atoms_refined or len(working_atoms)

    def calculate_r_based(self) -> Optional[DPIResult]:
        p = self.params
        if p.r_work is None or p.resolution is None or p.n_obs is None:
            return None
        working = self._working_atoms()
        n_a = self._n_atoms(working)
        n_params = p.n_params if p.n_params is not None else 4 * n_a
        dof = p.n_obs - n_params
        if dof <= 0:
            return None
        comp = p.completeness or 1.0
        sigma_x = math.sqrt(n_a / dof) * (comp ** (-1/3)) * p.r_work * p.resolution
        sigma_r = math.sqrt(3) * sigma_x
        b_avg = self._b_average(working)
        z_avg = self._z_average(working)
        output = self._output_atoms()
        per_atom = self._per_atom(output, sigma_x, b_avg, z_avg)
        return DPIResult(
            params=p, method='R', sigma_x_avg=sigma_x, sigma_r_avg=sigma_r,
            p=dof, b_avg=b_avg, z_avg=z_avg,
            n_atoms_used=len(working), per_atom=per_atom,
        )

    def calculate_rfree_based(self) -> Optional[DPIResult]:
        p = self.params
        if p.r_free is None or p.resolution is None or p.n_obs is None:
            return None
        working = self._working_atoms()
        n_a = p.n_atoms_refined or len(working)
        comp = p.completeness or 1.0
        sigma_x = math.sqrt(n_a / p.n_obs) * (comp ** (-1/3)) * p.r_free * p.resolution
        sigma_r = math.sqrt(3) * sigma_x
        b_avg = self._b_average(working)
        z_avg = self._z_average(working)
        output = self._output_atoms()
        per_atom = self._per_atom(output, sigma_x, b_avg, z_avg)
        return DPIResult(
            params=p, method='Rfree', sigma_x_avg=sigma_x, sigma_r_avg=sigma_r,
            p=None, b_avg=b_avg, z_avg=z_avg,
            n_atoms_used=len(working), per_atom=per_atom,
        )

    def _per_atom(
        self, atoms: List[Atom], sigma_x_avg: float, b_avg: float, z_avg: float
    ) -> List[AtomDPI]:
        result = []
        for atom in atoms:
            if b_avg > 0 and atom.b_iso > 0:
                b_corr = math.sqrt(atom.b_iso / b_avg)
            else:
                b_corr = 1.0
            if self.apply_z_correction:
                z_corr = z_avg / max(atom.atomic_number, 1)
            else:
                z_corr = 1.0
            sigma_x_i = sigma_x_avg * z_corr * b_corr
            sigma_r_i = math.sqrt(3) * sigma_x_i
            result.append(AtomDPI(atom=atom, sigma_x=sigma_x_i, sigma_r=sigma_r_i))
        return result


# ---------------------------------------------------------------------------
# Entry point for Pyodide
# ---------------------------------------------------------------------------

def calculate_from_file(file_content: str, filename: str, overrides: dict) -> dict:
    """Parse a PDB or mmCIF file and compute DPI. Returns a JSON-serializable dict.

    Parameters
    ----------
    file_content : str
        Raw text content of the uploaded file.
    filename : str
        Original filename (used to detect format via extension).
    overrides : dict
        Optional manual parameter overrides (keys: resolution, r_work, r_free,
        completeness, n_obs, n_params, n_atoms).

    Returns
    -------
    dict with keys:
        success (bool), error (str|None), params (dict), r_result (dict|None),
        rfree_result (dict|None), per_atom (list of dicts)
    """
    try:
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext in ('cif', 'mmcif'):
            atoms, params = MMCIFParser.parse(file_content)
        elif ext == 'pdb':
            atoms, params = PDBParser.parse(file_content)
        else:
            # Try mmCIF first (has 'data_' header), then PDB
            if file_content.lstrip().startswith('data_'):
                atoms, params = MMCIFParser.parse(file_content)
            else:
                atoms, params = PDBParser.parse(file_content)

        if not atoms:
            return {'success': False, 'error': 'No atoms found in file. Check the file format.'}

        # Apply overrides
        def _maybe_float(key):
            v = overrides.get(key)
            if v is not None and str(v).strip():
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
            return None

        def _maybe_int(key):
            v = overrides.get(key)
            if v is not None and str(v).strip():
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    pass
            return None

        ov_res = _maybe_float('resolution')
        ov_r_work = _maybe_float('r_work')
        ov_r_free = _maybe_float('r_free')
        ov_comp = _maybe_float('completeness')
        ov_n_obs = _maybe_int('n_obs')
        ov_n_params = _maybe_int('n_params')
        ov_n_atoms = _maybe_int('n_atoms')

        if ov_res is not None:
            params.resolution = ov_res
        if ov_r_work is not None:
            params.r_work = ov_r_work if ov_r_work < 1 else ov_r_work / 100.0
        if ov_r_free is not None:
            params.r_free = ov_r_free if ov_r_free < 1 else ov_r_free / 100.0
        if ov_comp is not None:
            params.completeness = ov_comp if ov_comp <= 1 else ov_comp / 100.0
        if ov_n_obs is not None:
            params.n_obs = ov_n_obs
        if ov_n_params is not None:
            params.n_params = ov_n_params
        if ov_n_atoms is not None:
            params.n_atoms_refined = ov_n_atoms

        include_hetatm = overrides.get('include_hetatm', True)
        # Pyodide JS→Python conversion may deliver booleans as strings
        if isinstance(include_hetatm, str):
            include_hetatm = include_hetatm.lower() in ('true', '1', 'yes')

        calc = DPICalculator(atoms, params, include_hetatm=include_hetatm, apply_z_correction=False)
        r_result = calc.calculate_r_based()
        rfree_result = calc.calculate_rfree_based()

        def _result_to_dict(res):
            if res is None:
                return None
            return {
                'method': res.method,
                'sigma_x_avg': round(res.sigma_x_avg, 4),
                'sigma_r_avg': round(res.sigma_r_avg, 4),
                'p': res.p,
                'b_avg': round(res.b_avg, 3),
                'z_avg': round(res.z_avg, 3),
                'n_atoms_used': res.n_atoms_used,
            }

        per_atom_rows = []
        # Use r_result atoms if available, otherwise rfree_result
        result_with_atoms = r_result or rfree_result
        if result_with_atoms:
            for ad in result_with_atoms.per_atom:
                a = ad.atom
                per_atom_rows.append({
                    'serial': a.serial,
                    'chain': a.chain_id,
                    'res_seq': a.res_seq,
                    'ins_code': a.ins_code,
                    'res_name': a.res_name,
                    'atom_name': a.name,
                    'alt_loc': a.alt_loc,
                    'element': a.element,
                    'occupancy': round(a.occupancy, 2),
                    'b_iso': round(a.b_iso, 2),
                    'sigma_x': round(ad.sigma_x, 4),
                    'sigma_r': round(ad.sigma_r, 4),
                })

        params_dict = {
            'source': params.source,
            'resolution': params.resolution,
            'r_work': params.r_work,
            'r_free': params.r_free,
            'completeness': params.completeness,
            'n_obs': params.n_obs,
            'n_params': params.n_params,
            'n_atoms_refined': params.n_atoms_refined,
            'space_group': params.space_group,
            'n_atoms_parsed': len(atoms),
        }

        return {
            'success': True,
            'error': None,
            'params': params_dict,
            'r_result': _result_to_dict(r_result),
            'rfree_result': _result_to_dict(rfree_result),
            'per_atom': per_atom_rows,
        }

    except Exception as exc:
        return {'success': False, 'error': str(exc)}
