#!/usr/bin/env python3
import mdtraj as md
import numpy as np
from Bio import Align
from Bio.Data import IUPACData

# =========================
# USER INPUTS
# =========================
XTC_FILE = "samples_sidechain_rec.xtc"
TOPOLOGY_PDB = "samples_sidechain_rec.pdb"

# State references (already aligned to COMMON_REF_PDB)
ACTIVE_PDB   = "7l1u_aligned.pdb"
INACTIVE_PDB = "4s0v_aligned.pdb"

# TM helix residue ranges (example: list of residue numbers in reference PDB)
# Replace with your receptor's TM residues
TM_RESIDUES_ACTIVE   = list(range(55, 81)) + list(range(90, 117)) + list(range(124, 157)) + list(range(166, 188)) + list(range(219, 249)) + list(range(294, 328)) + list(range(339, 367))
TM_RESIDUES_INACTIVE = list(range(55, 81)) + list(range(90, 117)) + list(range(124, 157)) + list(range(166, 188)) + list(range(219, 249)) + list(range(294, 328)) + list(range(339, 367))

ATOM_NAME = "CA"  # robust for TM mapping

# =========================
# Helper functions
# =========================
# Three-letter to one-letter code mapping
three_to_one = IUPACData.protein_letters_3to1.copy()
three_to_one.update({'SEC':'U','PYL':'O'})

def three_to_one_seq(res_list):
    return ''.join([three_to_one.get(r,'X') for r in res_list])

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

def map_tm_residues_to_indices(ref_traj, tm_residues):
    """Return CA atom indices corresponding to TM residues in the reference."""
    indices = []
    for res in ref_traj.topology.residues:
        if res.is_protein and res.resSeq in tm_residues:
            try:
                ca = res.atom(ATOM_NAME)
                indices.append(ca.index)
            except KeyError:
                continue
    return indices

def build_atom_index_mapping(traj, ref):
    """Build index mapping between trajectory and reference using CA sequence alignment."""
    traj_seq_3, traj_ca = extract_sequence_and_ca_indices(traj)
    ref_seq_3, ref_ca    = extract_sequence_and_ca_indices(ref)

    traj_seq_str = three_to_one_seq(traj_seq_3)
    ref_seq_str  = three_to_one_seq(ref_seq_3)

    # Use modern PairwiseAligner
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignment = aligner.align(ref_seq_str, traj_seq_str)[0]

    aln_ref, aln_traj = alignment.aligned
    traj_indices = []
    ref_indices  = []

    for (r_start, r_end), (t_start, t_end) in zip(aln_ref, aln_traj):
        for r_i, t_i in zip(range(r_start, r_end), range(t_start, t_end)):
            traj_indices.append(traj_ca[t_i])
            ref_indices.append(ref_ca[r_i])

    if len(traj_indices) < 10:
        raise ValueError("Too few matched atoms for RMSD")
    return traj_indices, ref_indices

# =========================
# Load data
# =========================
print("Loading trajectory...")
traj = md.load_xtc(XTC_FILE, top=TOPOLOGY_PDB)

print("Loading references...")
active_ref = md.load_pdb(ACTIVE_PDB)
inactive_ref = md.load_pdb(INACTIVE_PDB)

# Map TM residues to CA atom indices
tm_active_idx   = map_tm_residues_to_indices(active_ref, TM_RESIDUES_ACTIVE)
tm_inactive_idx = map_tm_residues_to_indices(inactive_ref, TM_RESIDUES_INACTIVE)

# =========================
# Align trajectory to active, compute RMSD vs active
# =========================
print("Aligning trajectory to active state...")
traj_aligned_active = traj.superpose(active_ref, atom_indices=tm_active_idx)
rmsd_active = md.rmsd(traj_aligned_active, active_ref, atom_indices=tm_active_idx)

# =========================
# Align trajectory to inactive, compute RMSD vs inactive
# =========================
print("Aligning trajectory to inactive state...")
traj_aligned_inactive = traj.superpose(inactive_ref, atom_indices=tm_inactive_idx)
rmsd_inactive = md.rmsd(traj_aligned_inactive, inactive_ref, atom_indices=tm_inactive_idx)

# =========================
# Output
# =========================
np.savetxt(
    "rmsd_tm_per_state.dat",
    np.column_stack([rmsd_active, rmsd_inactive]),
    header="RMSD_active  RMSD_inactive"
)

print("Frame   RMSD_active   RMSD_inactive  ΔRMSD")
for i, (ra, ri) in enumerate(zip(rmsd_active, rmsd_inactive)):
    delta = ri - ra
    print(f"{i:5d}   {ra:12.4f}   {ri:14.4f}   {delta:10.4f}")

print("\nDone.")

print("Number of TM CA atoms (active ref):", len(tm_active_idx))
print("Number of TM CA atoms (inactive ref):", len(tm_inactive_idx))
