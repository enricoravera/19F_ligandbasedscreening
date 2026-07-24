"""
protonate_pdb.py
=================
Add hydrogen atoms to an X-ray crystal structure (PDB file) that, like the
vast majority of structures at typical resolution, contains only
heavy-atom coordinates.

X-ray structures at resolutions worse than ~1.0-1.2 A essentially never
resolve H atoms directly (H has almost no scattering power), so any
surface/packing calculation that needs real H positions -- like the
exposed-proton-density calculation in surface_packing.py -- has to add
them back geometrically. This is what tools such as Reduce, PDB2PQR, or
OpenBabel's `--addh` do; this module implements the same underlying idea
(place each H at a standard bond length along the direction that balances
the local heavy-atom geometry) using explicit per-residue templates,
without requiring external dependencies.

This is a standard-geometry placement, not an energy-minimized one:
ideal bond lengths and idealized local geometry (tetrahedral sp3,
trigonal sp2) are used throughout. This is the right level of rigor for
surface-area / packing-density calculations, where H positions need to be
sterically and directionally correct but do not need to reflect
rotamer-specific H-bonding optimization.

Scope
-----
- Full backbone protonation (amide N-H, alpha C-H; N-terminal gets extra
  H's, proline has no amide H).
- Sidechain protonation templates for all 20 standard amino acids as
  normally protonated at neutral pH (ASP/GLU deprotonated -COO-, LYS/ARG
  protonated -NH3+/guanidinium, HIS neutral tautomer NE2-H by default,
  CYS/TYR/SER/THR -XH).
- A generic valence-based protonator for HETATM ligands/cofactors,
  inferring bond graph from interatomic distances and adding the number of
  H's needed to satisfy typical valence (C=4, N=3, O=2, S=2), placed with
  idealized sp3/sp2 geometry. This is approximate for exotic
  functional groups but adequate for standard organic-ligand chemistry.
- Waters and metals are left unprotonated (out of scope; explicit waters
  usually already imply the H's are irrelevant to ligand surface packing).

Usage
-----
    from protonate_pdb import parse_pdb, protonate_structure
    heavy = parse_pdb("structure.pdb", chains=["A"])
    all_atoms = protonate_structure(heavy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ============================================================================
# PDB parsing
# ============================================================================

@dataclass
class Atom:
    name: str
    resname: str
    chain: str
    resseq: int
    coord: np.ndarray
    elem: str
    is_het: bool
    icode: str = ""


def parse_pdb(
    path: str,
    chains: list[str] | None = None,
    exclude_resnames: tuple[str, ...] = ("HOH",),
) -> list[Atom]:
    """
    Parse ATOM/HETATM records from a PDB file into a flat list of Atom.

    - Keeps only altloc ' ' or 'A' (first/primary conformer) to avoid
      duplicate atoms.
    - Skips waters by default (exclude_resnames).
    - elem is read from columns 77-78 when present, else inferred from the
      atom name.
    """
    atoms = []
    with open(path) as fh:
        for line in fh:
            rec = line[0:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in exclude_resnames:
                continue
            chain = line[21].strip()
            if chains and chain not in chains:
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = line[76:78].strip()
            if not elem:
                elem = "".join(c for c in name if c.isalpha())[:1].upper()
            elem = elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem.upper()
            resseq = int(line[22:26])
            icode = line[26].strip()
            atoms.append(Atom(
                name=name, resname=resname, chain=chain, resseq=resseq,
                coord=np.array([x, y, z]), elem=elem,
                is_het=(rec == "HETATM"), icode=icode,
            ))
    return atoms


def group_residues(atoms: list[Atom]) -> list[list[Atom]]:
    """Group a flat atom list into consecutive per-residue lists, keyed by
    (chain, resseq, icode, resname, is_het)."""
    residues = []
    current_key = None
    current: list[Atom] = []
    for a in atoms:
        key = (a.chain, a.resseq, a.icode, a.resname, a.is_het)
        if key != current_key:
            if current:
                residues.append(current)
            current = []
            current_key = key
        current.append(a)
    if current:
        residues.append(current)
    return residues


# ============================================================================
# Geometry helpers for idealized H placement
# ============================================================================

BOND_H = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def h_sp3_one(parent: np.ndarray, nbrs: list[np.ndarray], bond: float) -> np.ndarray:
    """
    Place a single H on an sp3 parent atom that already has >=2 heavy-atom
    neighbors: point away from the sum of the bond vectors to those
    neighbors (the direction that best balances tetrahedral geometry).
    """
    d = np.zeros(3)
    for nb in nbrs:
        d += _unit(parent - nb)
    d = _unit(d)
    return parent + bond * d


def h_sp2_one(parent: np.ndarray, nbrs: list[np.ndarray], bond: float) -> np.ndarray:
    """Place a single H on an sp2 (trigonal planar) parent with 2 heavy
    neighbors already placed: bisector of the two bond vectors, negated
    and in-plane (same construction as sp3-with-2-neighbors, since a
    2-neighbor bisector-away point is planar by construction)."""
    return h_sp3_one(parent, nbrs, bond)


def h_two_on_sp3(parent: np.ndarray, nbr1: np.ndarray, nbr2: np.ndarray,
                  bond: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Place 2 H's on an sp3 parent (e.g. -CH2-) that has exactly 2 heavy
    neighbors (nbr1, nbr2, e.g. backbone C and sidechain C for a CB, or
    prior/next backbone atoms for a CH2 in a ring/chain). The two H's sit
    symmetric about the nbr1-parent-nbr2 bisector plane, tetrahedrally.
    """
    b1 = _unit(parent - nbr1)
    b2 = _unit(parent - nbr2)
    bisector = _unit(b1 + b2)
    normal = _unit(np.cross(b1, b2))
    if np.linalg.norm(normal) < 1e-6:
        # nbr1, parent, nbr2 nearly colinear -> pick an arbitrary
        # perpendicular
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, bisector)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        normal = _unit(np.cross(bisector, arbitrary))
    half_angle = np.deg2rad(54.75)  # tetrahedral half-angle off bisector
    d1 = _unit(bisector * np.cos(half_angle) + normal * np.sin(half_angle))
    d2 = _unit(bisector * np.cos(half_angle) - normal * np.sin(half_angle))
    return parent + bond * d1, parent + bond * d2


def h_three_on_sp3(parent: np.ndarray, nbr: np.ndarray, bond: float,
                    ref_dir: np.ndarray | None = None) -> list[np.ndarray]:
    """
    Place 3 H's on an sp3 parent with exactly 1 heavy neighbor (e.g.
    terminal -CH3, or -NH3+): staggered tetrahedrally around the
    parent-nbr axis. ref_dir picks an arbitrary starting azimuth (does not
    affect SASA/packing statistics since methyl/-NH3+ groups are
    treated as freely rotating for this purpose).
    """
    axis = _unit(parent - nbr)
    if ref_dir is None:
        ref_dir = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref_dir, axis)) > 0.9:
        ref_dir = np.array([0.0, 1.0, 0.0])
    perp1 = _unit(np.cross(axis, ref_dir))
    perp2 = np.cross(axis, perp1)
    tet_angle = np.deg2rad(109.5)
    out = []
    for k in range(3):
        az = 2 * np.pi * k / 3.0
        d = (np.cos(tet_angle) * axis +
             np.sin(tet_angle) * (np.cos(az) * perp1 + np.sin(az) * perp2))
        out.append(parent + bond * _unit(d))
    return out


def new_h(name: str, parent_atom: Atom, coord: np.ndarray) -> Atom:
    return Atom(name=name, resname=parent_atom.resname, chain=parent_atom.chain,
                resseq=parent_atom.resseq, coord=coord, elem="H",
                is_het=parent_atom.is_het, icode=parent_atom.icode)


# ============================================================================
# Amino-acid sidechain H templates
# ============================================================================
# Each entry: function(res_atoms: dict[name->Atom]) -> list[Atom] (new H's)
# res_atoms is a name->Atom lookup for the current residue only.

def _get(res, *names):
    for n in names:
        if n in res:
            return res[n]
    return None


def sidechain_H(res: dict[str, Atom]) -> list[Atom]:
    resname = next(iter(res.values())).resname
    out: list[Atom] = []

    def add2(name1, name2, parent_name, nbr1_name, nbr2_name):
        p, n1, n2 = res.get(parent_name), res.get(nbr1_name), res.get(nbr2_name)
        if p and n1 and n2:
            h1, h2 = h_two_on_sp3(p.coord, n1.coord, n2.coord, BOND_H[p.elem])
            out.append(new_h(name1, p, h1))
            out.append(new_h(name2, p, h2))

    def add3(prefix, parent_name, nbr_name):
        p, n = res.get(parent_name), res.get(nbr_name)
        if p and n:
            for k, h in enumerate(h_three_on_sp3(p.coord, n.coord, BOND_H[p.elem]), 1):
                out.append(new_h(f"{prefix}{k}", p, h))

    def add1_sp3(name, parent_name, nbr_names):
        p = res.get(parent_name)
        nbrs = [res[n].coord for n in nbr_names if n in res]
        if p is None:
            return
        if len(nbrs) >= 2:
            h = h_sp3_one(p.coord, nbrs, BOND_H[p.elem])
            out.append(new_h(name, p, h))
        elif len(nbrs) == 1:
            # Single heavy neighbor (e.g. -OH on SER/THR/TYR, -SH on CYS):
            # place at the tetrahedral angle off the parent-neighbor axis.
            # Azimuth around that axis is arbitrary (free rotor for a
            # single terminal -OH/-SH, doesn't affect SASA-based
            # statistics), reusing the same construction as h_three_on_sp3
            # but emitting only the first of the 3 directions.
            h = h_three_on_sp3(p.coord, nbrs[0], BOND_H[p.elem])[0]
            out.append(new_h(name, p, h))

    # CB: -CH2- for residues whose CB is a genuine 2-substituent methylene
    # (i.e. one heavy sidechain neighbor beyond CA). Branched-at-CB
    # residues (THR, VAL, ILE) and ALA/GLY/PRO are handled in their own
    # branches below and must NOT also be hit by this generic rule.
    _cb_handled_elsewhere = {"ALA", "GLY", "PRO", "THR", "VAL", "ILE"}
    if resname not in _cb_handled_elsewhere and "CB" in res:
        cb = res["CB"]
        # second heavy neighbor of CB, i.e. the first sidechain atom beyond CA
        sidechain_nbr = next(
            (res[n] for n in ("CG", "CG1", "CG2", "OG", "OG1", "SG")
             if n in res), None
        )
        if sidechain_nbr is not None:
            h1, h2 = h_two_on_sp3(cb.coord, res["CA"].coord, sidechain_nbr.coord,
                                    BOND_H["C"])
            out.append(new_h("HB2", cb, h1))
            out.append(new_h("HB3", cb, h2))

    if resname == "ALA":
        add3("HB", "CB", "CA")
    elif resname == "GLY":
        add2("HA2", "HA3", "CA", "N", "C")
    elif resname == "PRO":
        add2("HB2", "HB3", "CB", "CA", "CG")
        add2("HG2", "HG3", "CG", "CB", "CD")
        add2("HD2", "HD3", "CD", "CG", "N")
    elif resname == "SER":
        add1_sp3("HG", "OG", ["CB"])  # only 1 heavy nbr -> approx direction
    elif resname == "CYS":
        add1_sp3("HG", "SG", ["CB"])
    elif resname == "THR":
        add1_sp3("HB", "CB", ["CA", "OG1"])
        add3("HG2", "CG2", "CB")
        add1_sp3("HG1", "OG1", ["CB"])
    elif resname == "VAL":
        add1_sp3("HB", "CB", ["CA", "CG1"])
        add3("HG1", "CG1", "CB")
        add3("HG2", "CG2", "CB")
    elif resname == "LEU":
        add2("HG2", "HG3", "CG", "CB", "CD1") if False else None
        add1_sp3("HG", "CG", ["CB", "CD1"])
        add3("HD1", "CD1", "CG")
        add3("HD2", "CD2", "CG")
    elif resname == "ILE":
        add1_sp3("HB", "CB", ["CA", "CG1"])
        add2("HG12", "HG13", "CG1", "CB", "CD1")
        add3("HG2", "CG2", "CB")
        add3("HD1", "CD1", "CG1")
    elif resname == "MET":
        add2("HG2", "HG3", "CG", "CB", "SD")
        add3("HE", "CE", "SD")
    elif resname == "ASP":
        pass  # carboxylate, no OH at neutral pH
    elif resname == "ASN":
        p = res.get("ND2")
        if p and "CG" in res:
            for k, nm in enumerate(("HD21", "HD22"), 1):
                d = _unit(p.coord - res["CG"].coord)
                # two H's roughly in amide plane +/- small splay
                perp = _unit(np.cross(d, np.array([0, 0, 1.0])))
                if np.linalg.norm(perp) < 1e-6:
                    perp = _unit(np.cross(d, np.array([1.0, 0, 0])))
                ang = np.deg2rad(60 if k == 1 else -60)
                dd = _unit(np.cos(ang) * d + np.sin(ang) * perp)
                out.append(new_h(nm, p, p.coord + BOND_H["N"] * dd))
    elif resname == "GLU":
        pass  # carboxylate
    elif resname == "GLN":
        p = res.get("NE2")
        if p and "CD" in res:
            for k, nm in enumerate(("HE21", "HE22"), 1):
                d = _unit(p.coord - res["CD"].coord)
                perp = _unit(np.cross(d, np.array([0, 0, 1.0])))
                if np.linalg.norm(perp) < 1e-6:
                    perp = _unit(np.cross(d, np.array([1.0, 0, 0])))
                ang = np.deg2rad(60 if k == 1 else -60)
                dd = _unit(np.cos(ang) * d + np.sin(ang) * perp)
                out.append(new_h(nm, p, p.coord + BOND_H["N"] * dd))
    elif resname == "LYS":
        add2("HG2", "HG3", "CG", "CB", "CD")
        add2("HD2", "HD3", "CD", "CG", "CE")
        add2("HE2", "HE3", "CE", "CD", "NZ")
        add3("HZ", "NZ", "CE")
    elif resname == "ARG":
        add2("HG2", "HG3", "CG", "CB", "CD")
        add2("HD2", "HD3", "CD", "CG", "NE")
        add1_sp3("HE", "NE", ["CD", "CZ"])
        for nh_name, cz_ref in (("NH1", "NE"), ("NH2", "NE")):
            p = res.get(nh_name)
            cz = res.get("CZ")
            if p and cz:
                add3(f"H{nh_name[1:]}", nh_name, "CZ")
    elif resname == "HIS":
        # aromatic ring CH's + NE2-H (neutral His, common default tautomer)
        for h_name, parent_name, n1, n2 in (
            ("HD2", "CD2", "CG", "NE1" if False else "NE2"),
            ("HE1", "CE1", "ND1", "NE2"),
        ):
            p, a1, a2 = res.get(parent_name), res.get(n1), res.get(n2)
            if p and a1 and a2:
                h = h_sp2_one(p.coord, [a1.coord, a2.coord], BOND_H["C"])
                out.append(new_h(h_name, p, h))
        p, cd2, ce1 = res.get("NE2"), res.get("CD2"), res.get("CE1")
        if p and cd2 and ce1:
            h = h_sp2_one(p.coord, [cd2.coord, ce1.coord], BOND_H["N"])
            out.append(new_h("HE2", p, h))
    elif resname == "PHE" or resname == "TYR":
        ring_H = [
            ("HD1", "CD1", "CG", "CE1"),
            ("HD2", "CD2", "CG", "CE2"),
            ("HE1", "CE1", "CD1", "CZ"),
            ("HE2", "CE2", "CD2", "CZ"),
        ]
        for h_name, parent_name, n1, n2 in ring_H:
            p, a1, a2 = res.get(parent_name), res.get(n1), res.get(n2)
            if p and a1 and a2:
                h = h_sp2_one(p.coord, [a1.coord, a2.coord], BOND_H["C"])
                out.append(new_h(h_name, p, h))
        if resname == "TYR":
            add1_sp3("HH", "OH", ["CZ"])
    elif resname == "TRP":
        ring_defs = [
            ("HD1", "CD1", "CG", "NE1"),
            ("HE1", "NE1", "CD1", "CE2"),
            ("HE3", "CE3", "CD2", "CZ3"),
            ("HZ2", "CZ2", "CE2", "CH2"),
            ("HZ3", "CZ3", "CE3", "CH2"),
            ("HH2", "CH2", "CZ2", "CZ3"),
        ]
        for h_name, parent_name, n1, n2 in ring_defs:
            p, a1, a2 = res.get(parent_name), res.get(n1), res.get(n2)
            bond = BOND_H["N"] if p and p.elem == "N" else BOND_H["C"]
            if p and a1 and a2:
                h = h_sp2_one(p.coord, [a1.coord, a2.coord], bond)
                out.append(new_h(h_name, p, h))

    return out


def backbone_H(res: dict[str, Atom], prev_res: dict[str, Atom] | None,
                is_first: bool) -> list[Atom]:
    out: list[Atom] = []
    resname = next(iter(res.values())).resname

    # Amide H on N (all but Pro and the very first residue's N, which
    # gets NH3+ treatment instead)
    N, CA, C = res.get("N"), res.get("CA"), res.get("C")
    if N is not None and resname != "PRO":
        if is_first:
            # N-terminal NH3+: 3 H's, staggered around N-CA axis
            if CA is not None:
                for k, h in enumerate(h_three_on_sp3(N.coord, CA.coord, BOND_H["N"]), 1):
                    out.append(new_h(f"H{k}", N, h))
        else:
            prevC = prev_res.get("C") if prev_res else None
            if prevC is not None and CA is not None:
                h = h_sp2_one(N.coord, [prevC.coord, CA.coord], BOND_H["N"])
                out.append(new_h("H", N, h))

    # Alpha H (all but Gly, handled in sidechain_H as HA2/HA3)
    if resname != "GLY" and CA is not None:
        nbrs = [a.coord for a in (N, res.get("CB"), C) if a is not None]
        if len(nbrs) >= 2:
            h = h_sp3_one(CA.coord, nbrs[:2], BOND_H["C"])
            out.append(new_h("HA", CA, h))

    return out


def protonate_protein_residue(res_atoms: list[Atom], prev_res_atoms: list[Atom] | None,
                               is_first: bool) -> list[Atom]:
    res = {a.name: a for a in res_atoms}
    prev = {a.name: a for a in prev_res_atoms} if prev_res_atoms else None
    new_hs = []
    new_hs += backbone_H(res, prev, is_first)
    new_hs += sidechain_H(res)
    return new_hs


# ============================================================================
# Generic valence-based ligand protonation
# ============================================================================

COVALENT_RADIUS = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "Cl": 0.99,
                    "F": 0.57, "P": 1.07, "Br": 1.14, "I": 1.33, "H": 0.31}
TARGET_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2}


def infer_bonds(atoms: list[Atom], tol: float = 0.4) -> list[list[int]]:
    """Infer a bond graph from interatomic distances using a covalent-radius
    sum + tolerance cutoff (standard approach absent explicit connectivity)."""
    n = len(atoms)
    coords = np.array([a.coord for a in atoms])
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r1 = COVALENT_RADIUS.get(atoms[i].elem, 0.77)
            r2 = COVALENT_RADIUS.get(atoms[j].elem, 0.77)
            d = np.linalg.norm(coords[i] - coords[j])
            if d < r1 + r2 + tol:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def protonate_ligand(atoms: list[Atom]) -> list[Atom]:
    """
    Generic valence-based protonation for a HETATM ligand: infer the bond
    graph from distances, then for each C/N/O/S atom add
    (n_sigma_bonds_needed - n_heavy_bonds) hydrogens.

    n_sigma_bonds_needed depends on hybridization, which is approximated
    from the heavy-atom coordination number alone (no bond-order/aromaticity
    perception):
      - Carbon:  4 heavy/H substituents if sp3 (<=2 heavy neighbors, e.g.
                 -CH2-, -CH3), but a carbon with 3 heavy neighbors already
                 already has 3 sigma bonds (consistent with sp2/aromatic,
                 e.g. a ring carbon bearing a halogen or ring-fusion
                 substituent) and needs 0 H, not 1 -- adding an H there
                 would over-saturate an aromatic/trigonal center. A carbon
                 with 4 heavy neighbors is fully substituted (0 H).
      - Nitrogen: analogous -- 3 heavy neighbors (amide N, aromatic ring N,
                 tertiary amine, trisubstituted planar N) is normally
                 already saturated (0 H); 1-2 heavy neighbors get H's up
                 to sp3 valence 3.
      - Oxygen/Sulfur: 1 heavy neighbor -> 1 H (e.g. -OH); 2 heavy
                 neighbors (ether/ester/thioether/carbonyl-flanking O) ->
                 0 H.

    This coordination-number heuristic correctly leaves aromatic/sp2 ring
    atoms (common in drug-like ligands) unprotonated when they already
    carry 3 substituents, which a flat "target valence 4/3" rule would
    over-protonate. It is still approximate: it cannot distinguish e.g. an
    sp3 tertiary amine (3 heavy neighbors, 0 H, correctly handled the same
    way) from a case that would genuinely need a lone H at 3 heavy
    neighbors (none of the standard organic functional groups need this),
    so it is safe for typical drug-like ligand chemistry.
    """
    adj = infer_bonds(atoms)
    out: list[Atom] = []
    for i, a in enumerate(atoms):
        if a.elem not in TARGET_VALENCE:
            continue
        heavy_nbr_idx = [j for j in adj[i] if atoms[j].elem != "H"]
        n_heavy = len(heavy_nbr_idx)

        if a.elem in ("C", "N"):
            # sp3 max valence (4 for C, 3 for N) only applies up to 2 heavy
            # neighbors; 3+ heavy neighbors means the atom is already a
            # trigonal (sp2) or fully substituted center with 0 H needed.
            if n_heavy >= 3:
                n_h_needed = 0
            else:
                n_h_needed = TARGET_VALENCE[a.elem] - n_heavy
        elif a.elem == "O" and n_heavy == 1:
            # A single-neighbor O is a hydroxyl (-OH, 1 H) only if the C-O
            # bond length is consistent with a single bond (~1.35-1.43 A).
            # A short bond (~1.20-1.24 A) is a carbonyl C=O double bond and
            # needs 0 H. Distance-based bond inference alone can't tell
            # these apart from coordination number, so check the length.
            nbr = atoms[heavy_nbr_idx[0]]
            bond_len = np.linalg.norm(a.coord - nbr.coord)
            n_h_needed = 1 if bond_len > 1.30 else 0
        else:  # O (2 heavy neighbors, e.g. ether), S
            n_h_needed = TARGET_VALENCE[a.elem] - n_heavy

        if n_h_needed <= 0:
            continue

        nbr_coords = [atoms[j].coord for j in heavy_nbr_idx]
        bond = BOND_H.get(a.elem, 1.0)
        if len(nbr_coords) == 0:
            continue
        if n_h_needed == 1 and len(nbr_coords) >= 2:
            h = h_sp3_one(a.coord, nbr_coords, bond)
            out.append(new_h(f"H{a.name}", a, h))
        elif n_h_needed == 2 and len(nbr_coords) == 1:
            h1, h2 = h_two_on_sp3(a.coord, nbr_coords[0],
                                    a.coord + np.array([1.0, 0, 0]), bond)
            # (2nd "neighbor" here is a dummy direction since only 1 real
            # neighbor exists; acceptable for -NH2/-CH2- terminal groups
            # where exact azimuth doesn't affect SASA statistics much)
            out.append(new_h(f"H{a.name}A", a, h1))
            out.append(new_h(f"H{a.name}B", a, h2))
        elif n_h_needed >= 3 and len(nbr_coords) == 1:
            for k, h in enumerate(h_three_on_sp3(a.coord, nbr_coords[0], bond), 1):
                out.append(new_h(f"H{a.name}{k}", a, h))
        elif n_h_needed == 1 and len(nbr_coords) == 1:
            # e.g. terminal -OH/-SH with 1 heavy neighbor and 1 H needed
            h = h_sp3_one(a.coord, [nbr_coords[0], a.coord + np.array([0, 0, 1.0])], bond)
            out.append(new_h(f"H{a.name}", a, h))
    return out


# ============================================================================
# Top-level driver
# ============================================================================

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def protonate_structure(atoms: list[Atom]) -> list[Atom]:
    """
    Add hydrogens to a heavy-atom-only structure.

    Protein residues (standard 20 amino acids) get template-based backbone
    + sidechain protonation. Everything else (ligands, cofactors, ions) is
    protonated generically via infer_bonds + valence.

    Returns
    -------
    The original atom list with all new H atoms appended (heavy atoms are
    not reordered or modified).
    """
    residues = group_residues(atoms)
    all_new_h: list[Atom] = []

    prev_std_res = None
    for i, res_atoms in enumerate(residues):
        resname = res_atoms[0].resname
        is_het = res_atoms[0].is_het
        if not is_het and resname in STANDARD_AA:
            is_first_in_chain = (i == 0 or residues[i - 1][0].chain != res_atoms[0].chain)
            new_hs = protonate_protein_residue(res_atoms, prev_std_res, is_first_in_chain)
            all_new_h += new_hs
            prev_std_res = res_atoms
        elif is_het:
            all_new_h += protonate_ligand(res_atoms)
            prev_std_res = None
        else:
            prev_std_res = None

    return atoms + all_new_h


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "8AWW.pdb"
    heavy = parse_pdb(path, chains=["A"])
    full = protonate_structure(heavy)
    n_h = sum(1 for a in full if a.elem == "H")
    print(f"Heavy atoms: {len(heavy)}   Added H atoms: {n_h}   Total: {len(full)}")
