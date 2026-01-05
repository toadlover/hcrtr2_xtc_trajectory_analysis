#!/usr/bin/env python3

import mdtraj as md
import numpy as np
import os

# =========================
# USER INPUTS
# =========================
XTC_FILE = "samples_sidechain_rec.xtc"
TOPOLOGY_FILE = "samples_sidechain_rec.pdb"          # must match XTC atom order
REFERENCE_PDB = "af_aligned.pdb" # reference already aligned as desired

OUTPUT_DIR = "aligned_frames"
WRITE_PDBS = True      # True = write individual PDBs
WRITE_MULTI_PDB = False  # True = write one multi-model PDB

# Atom selection for alignment
ALIGN_SELECTION = "protein and backbone"

# =========================
# LOAD DATA
# =========================
print("Loading trajectory...")
traj = md.load_xtc(XTC_FILE, top=TOPOLOGY_FILE)

print("Loading reference...")
ref = md.load_pdb(REFERENCE_PDB)

# =========================
# SANITY CHECKS
# =========================
if traj.n_atoms != ref.n_atoms:
    raise ValueError("Atom count mismatch between trajectory and reference!")

print(f"Loaded {traj.n_frames} frames with {traj.n_atoms} atoms")

# =========================
# FIX PERIODIC BOUNDARIES
# =========================
print("Fixing periodic boundary conditions (if topology allows)...")
try:
    traj.image_molecules(inplace=True)
    traj.center_coordinates()
    print("PBC correction applied")
except Exception as e:
    print("WARNING: Could not fix PBC artifacts")
    print(f"Reason: {e}")

# =========================
# ALIGN TRAJECTORY
# =========================
print(f"Aligning trajectory using selection: '{ALIGN_SELECTION}'")
atom_indices = traj.topology.select(ALIGN_SELECTION)

if len(atom_indices) == 0:
    raise ValueError("Alignment selection returned zero atoms!")

traj.superpose(ref, atom_indices=atom_indices)
print("Alignment complete")

# =========================
# OPTIONAL RMSD CHECK
# =========================
rmsd = md.rmsd(traj, ref, atom_indices=atom_indices)
print(f"RMSD after alignment (first 10 frames):\n{rmsd[:10]}")

# =========================
# OUTPUT
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

if WRITE_MULTI_PDB:
    out_pdb = os.path.join(OUTPUT_DIR, "aligned_trajectory.pdb")
    traj.save_pdb(out_pdb)
    print(f"Wrote multi-model PDB: {out_pdb}")

if WRITE_PDBS:
    print("Writing individual frame PDBs...")
    for i, frame in enumerate(traj):
        frame_path = os.path.join(OUTPUT_DIR, f"frame_{i:05d}.pdb")
        frame.save_pdb(frame_path)

# =========================
# PER-FRAME COORDINATE ACCESS
# =========================
print("Extracting atom coordinates per frame (example)...")

for i, frame in enumerate(traj[:5]):  # only first 5 as example
    xyz = frame.xyz[0]  # shape: (n_atoms, 3)
    print(f"Frame {i}: xyz shape = {xyz.shape}")

print("Done.")
