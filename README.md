# pdb_dpi — Diffraction Precision Index Calculator

A Python script implementing the **Diffraction Precision Index (DPI)** for
macromolecular crystal structures, following Cruickshank (1999) with the
per-atom extension of Gurusaran *et al.* (2014).

---

## Background

The DPI gives an estimate of the coordinate precision of a refined crystal
structure without running a full covariance matrix inversion.  Two formulae
are provided:

**R-based** (Cruickshank 1999, Eq. 24)

```
σ(x, B_avg) = √(Nᵢ / p)     × C^(−1/3) × R      × d_min
```

**R_free-based** (Cruickshank 1999, Eq. 31)

```
σ(x, B_avg) = √(Nᵢ / n_obs) × C^(−1/3) × R_free × d_min
```

Isotropic position error:

```
σ(r, B_avg) = √3 × σ(x, B_avg)
```

**Per-atom precision** (Gurusaran *et al.* 2014, Eq. 6)

```
σ(x, Bᵢ) = σ(x, B_avg) × (Z_avg / Zᵢ) × exp[(Bᵢ − B_avg) / (2 × d_min²)]
```

| Symbol | Meaning |
|--------|---------|
| Nᵢ | Number of non-H atoms in refinement |
| p | N_obs − N_params (degrees of freedom) |
| n_obs | Total observed reflections used in refinement |
| C | Data completeness (0–1) |
| R | R_work (conventional R-factor) |
| R_free | Free R-factor |
| d_min | Maximum resolution (Å) |
| B_avg | Average B-factor of working atoms |
| Bᵢ | B-factor of atom *i* |
| Zᵢ | Atomic number of atom *i* |

> **Important:** The R_free formula uses **n_obs** (total observed reflections)
> in the denominator, **not** N_free (the free-set size).  This is explicit in
> Cruickshank (1999) Table 3, where the column heading for the R_free rows reads
> "(Nᵢ/n_obs)^½".

---

## Installation

No package installation is required — just Python 3.8+ with the standard library.

```bash
git clone https://github.com/LifeHasOrder/pdb_dpi.git
cd pdb_dpi
python dpi_calculator.py --help
```

To run the test suite:

```bash
pip install pytest
pytest test_dpi_calculator.py -v
```

---

## Usage

### Download from RCSB and compute DPI

```bash
python dpi_calculator.py --pdb-id 1NLS
```

### Use a local PDB or mmCIF file

```bash
python dpi_calculator.py --file my_structure.pdb
python dpi_calculator.py --file my_structure.cif
```

### Supply refinement statistics from a Phenix log

```bash
python dpi_calculator.py --file my_structure.pdb --phenix-log refine.log
```

### Override parsed parameters manually

```bash
python dpi_calculator.py --file my_structure.pdb \
    --resolution 1.5 --r-work 0.18 --r-free 0.22 \
    --n-obs 30000 --completeness 0.97
```

### Select which formula to use

```bash
# R-based only
python dpi_calculator.py --pdb-id 1NLS --method R

# R_free-based only
python dpi_calculator.py --pdb-id 1NLS --method Rfree
```

### Save per-atom CSV and annotated PDB

```bash
python dpi_calculator.py --pdb-id 1NLS --out-dir results/
# Writes: results/1NLS_dpi_r.csv, results/1NLS_dpi_rfree.csv
#         results/1NLS_dpi_r_annotated.pdb, …
```

### Full option list

```
python dpi_calculator.py --help
```

---

## Outputs

| File | Contents |
|------|----------|
| `<prefix>_dpi_r.csv` | Per-atom σ(x) and σ(r) using R-based formula |
| `<prefix>_dpi_rfree.csv` | Per-atom σ(x) and σ(r) using R_free-based formula |
| `<prefix>_dpi_r_annotated.pdb` | Input PDB with B-column replaced by σ(x) |
| `<prefix>_dpi_rfree_annotated.pdb` | Same with R_free-based values |

---

## Validation Against Cruickshank (1999) Table 3

The following table reproduces the published DPI values using only the
intermediate factors listed in the paper.  All values computed by this
script are within 15 % of the published results (rounding to 2 significant
figures accounts for most of the difference).

| Protein | Nᵢ | n_obs | (Nᵢ/p)^½ | C^(−1/3) | R | d_min (Å) | σ(r,R) published | σ(r,R) calc |
|---------|-----|-------|-----------|---------|-----|-----------|-----------------|-------------|
| Concanavalin A | 2130 | 116712 | 0.148 | 1.099 | 0.128 | 0.94 | 0.034 | 0.034 |
| HEW lysozyme (ground) | 1145 | 24111 | 0.242 | 1.048 | 0.184 | 1.33 | 0.11 | 0.107 |
| HEW lysozyme (space) | 1141 | 21542 | 0.259 | 1.040 | 0.183 | 1.40 | 0.12 | 0.120 |
| γB-crystallin | 1708 | 26151 | 0.297 | 1.032 | 0.180 | 1.49 | 0.14 | 0.142 |
| βB2-crystallin | 1558 | 18583 | 0.356 | 1.032 | 0.184 | 2.10 | 0.25 | 0.246 |
| β-purothionin | 439 | 4966 | 0.370 | 1.050 | 0.198 | 1.70 | 0.22 | 0.226 |
| EM lysozyme | 1068 | 8308 | 0.514 | 1.040 | 0.169 | 1.90 | 0.30 | 0.297 |
| Azurin II | 1012 | 12162 | 0.353 | 1.174 | 0.188 | 1.90 | 0.26 | 0.256 |
| Ribonuclease A+RI | 4416 | 18859 | 1.922 | 1.145 | 0.194 | 2.50 | 1.85 | 1.849 |

| Protein | (Nᵢ/n_obs)^½ | R_free | σ(r,Rfree) published | σ(r,Rfree) calc |
|---------|-------------|--------|---------------------|-----------------|
| Concanavalin A | 0.135 | 0.148 | 0.036 | 0.036 |
| HEW lysozyme (ground) | 0.218 | 0.226 | 0.12 | 0.119 |
| HEW lysozyme (space) | 0.230 | 0.226 | 0.13 | 0.131 |
| γB-crystallin | 0.256 | 0.204 | 0.14 | 0.139 |
| βB2-crystallin | 0.290 | 0.200 | 0.22 | 0.217 |
| β-purothionin | 0.297 | 0.281 | 0.26 | 0.258 |
| EM lysozyme | 0.359 | 0.229 | 0.28 | 0.281 |
| Azurin II | 0.288 | 0.207 | 0.23 | 0.231 |
| Ribonuclease A+RI | 0.484 | 0.286 | 0.69 | 0.686 |

---

## Notes and Caveats

### N_params auto-estimation

When N_params is not reported in the file (common for older PDB entries and
Phenix output), the script estimates it as **4 × N_atoms** (isotropic B,
i.e. *x, y, z, B* per atom).  Override with `--n-params` if you know the
true value.

### Alternate conformers

Only atoms with `alt_loc` = `''` or `'A'` are included by default to avoid
inflating Nᵢ and skewing B_avg.

### Completeness factor

If completeness is missing it defaults to 1.0 (C^(−1/3) = 1.0), which means
the DPI is slightly underestimated for incomplete datasets.

---

## References

1. Cruickshank, D. W. J. (1999). *Acta Cryst.* **D55**, 583–601.
   https://doi.org/10.1107/S0907444999001250
2. Blow, D. M. (2002). *Acta Cryst.* **D58**, 792–797.
3. Gurusaran, M. *et al.* (2014). *IUCrJ* **1**, 74–81.
   https://doi.org/10.1107/S2052252513031461
4. Kumar, K. S. D. *et al.* (2015). *J. Appl. Cryst.* **48**, 939–942.
5. Helliwell, J. R. (2023). *Curr. Res. Struct. Biol.* **6**, 100111.
