#!/usr/bin/env python3

import mdtraj as md
import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq

# =========================
# USER INPUTS
# =========================
XTC_FILE = "samples_sidechain_rec.xtc"
TOPOLOGY_PDB = "samples_sidechain_rec.pdb"

# Common reference used to define the coordinate frame
COMMON_REF_PDB = "af_aligned.pdb"

# State references (already aligned to COMMON_REF_PDB)
ACTIVE_PDB   = "7l1u_aligned.pdb"
INACTIVE_PDB = "4s0v_aligned.pdb"

ATOM_NAME = "CA"   # robust choice

# =========================
# HELPER FUNCTIONS
# =========================
def extract_sequence_and_ca_indices(traj):
    seq = []
    ca_indices = []

    for res in traj.topology.residues:
        if res.is_protein:
            try:
                ca = res.atom(ATOM_NAME)
                seq.append(res.name)
                ca_indices.append(ca.index)
            except KeyError:
                continue

    return seq, ca_indices


from Bio import Align
from Bio.Data import IUPACData

# build three-letter -> one-letter mapping
three_to_one = IUPACData.protein_letters_3to1.copy()
# add uncommon residues if present
three_to_one.update({
    'SEC': 'U',  # selenocysteine
    'PYL': 'O',  # pyrrolysine
})

def three_to_one_seq(res_list):
    """Convert a list of three-letter residue codes to a one-letter string."""
    seq_str = ''
    for r in res_list:
        seq_str += three_to_one.get(r, 'X')  # unknown residues -> 'X'
    return seq_str

def build_atom_index_mapping(traj, ref):
    """
    Build matching CA atom index lists for RMSD calculation
    using sequence alignment (Bio.Align.PairwiseAligner).

    Returns:
        traj_indices (list[int])
        ref_indices  (list[int])
    """
    # Extract sequences and CA atom indices
    traj_seq_3, traj_ca = extract_sequence_and_ca_indices(traj)
    ref_seq_3,  ref_ca  = extract_sequence_and_ca_indices(ref)

    traj_seq_str = three_to_one_seq(traj_seq_3)
    ref_seq_str  = three_to_one_seq(ref_seq_3)

    # Align sequences using PairwiseAligner
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignment = aligner.align(ref_seq_str, traj_seq_str)[0]  # get best alignment

    aln_ref, aln_traj = alignment.aligned
    # aln_ref and aln_traj are lists of (start, end) tuples for matched segments

    traj_indices = []
    ref_indices  = []

    for (r_start, r_end), (t_start, t_end) in zip(aln_ref, aln_traj):
        # map residues to CA atom indices
        for r_i, t_i in zip(range(r_start, r_end), range(t_start, t_end)):
            traj_indices.append(traj_ca[t_i])
            ref_indices.append(ref_ca[r_i])

    if len(traj_indices) < 10:
        raise ValueError("Too few matched CA atoms for RMSD calculation")

    return traj_indices, ref_indices



# =========================
# LOAD DATA
# =========================
print("Loading trajectory...")
traj = md.load_xtc(XTC_FILE, top=TOPOLOGY_PDB)

print("Loading common alignment reference...")
common_ref = md.load_pdb(COMMON_REF_PDB)

# =========================
# ALIGN TRAJECTORY FIRST
# =========================
print("Aligning trajectory to common reference...")

traj_idx_common, ref_idx_common = build_atom_index_mapping(traj, common_ref)

traj.superpose(
    common_ref,
    atom_indices=traj_idx_common,
    ref_atom_indices=ref_idx_common
)

print("Trajectory alignment complete")

# =========================
# LOAD STATE REFERENCES
# =========================
print("Loading state references...")
active_ref   = md.load_pdb(ACTIVE_PDB)
inactive_ref = md.load_pdb(INACTIVE_PDB)

# =========================
# BUILD RMSD MAPPINGS
# =========================
print("Building RMSD atom mappings...")

traj_idx_active,   ref_idx_active   = build_atom_index_mapping(traj, active_ref)
traj_idx_inactive, ref_idx_inactive = build_atom_index_mapping(traj, inactive_ref)

print(f"Matched CA atoms (active):   {len(traj_idx_active)}")
print(f"Matched CA atoms (inactive): {len(traj_idx_inactive)}")

# =========================
# COMPUTE RMSDs
# =========================
print("Computing RMSDs...")

rmsd_active = md.rmsd(
    traj,
    active_ref,
    atom_indices=traj_idx_active,
    ref_atom_indices=ref_idx_active
)

rmsd_inactive = md.rmsd(
    traj,
    inactive_ref,
    atom_indices=traj_idx_inactive,
    ref_atom_indices=ref_idx_inactive
)

# =========================
# OUTPUT
# =========================
np.savetxt(
    "rmsd_vs_states.dat",
    np.column_stack([rmsd_active, rmsd_inactive]),
    header="RMSD_active  RMSD_inactive"
)

print("Frame   RMSD_active   RMSD_inactive")
for i, (ra, ri) in enumerate(zip(rmsd_active, rmsd_inactive)):
    print(f"{i:5d}   {ra:12.4f}   {ri:14.4f}")

print("\nDone.")
