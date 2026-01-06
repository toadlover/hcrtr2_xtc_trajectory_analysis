#!/usr/bin/env python3
import mdtraj as md
import numpy as np
from Bio.Data import IUPACData
import pymol2
import os

# =========================
# USER INPUTS
# =========================
XTC_FILE = "samples_sidechain_rec.xtc"
TOPOLOGY_PDB = "samples_sidechain_rec.pdb"

# State references
ACTIVE_PDB   = "7l1u_aligned.pdb"
INACTIVE_PDB = "4s0v_aligned.pdb"

# TM residues for alignment (core residues)
TM1 = range(57, 78)
TM2 = range(92, 115)
TM3 = range(126, 155)
TM4 = range(168, 186)
TM5 = range(222, 246)
TM6 = range(296, 324)
TM7 = range(341, 364)
TM_RESIDUES = list(TM1) + list(TM2) + list(TM3) + list(TM4) + list(TM5) + list(TM6) + list(TM7)

# Binding pocket residues for RMSD (example, replace with your own list)
BINDING_POCKET = [61,103,106,107,108,109,110,111,112,113,114,115,120,122,127,129,130,131,132,133,134,135,136,138,142,184,187,191,210,211,212,223,224,227,228,231,232,313,317,320,321,323,324,325,328,346,347,349,350,351,353,354]

ATOM_NAME = "CA"

# =========================
# Helper functions
# =========================
three_to_one = IUPACData.protein_letters_3to1.copy()
three_to_one.update({'SEC':'U','PYL':'O'})

def three_to_one_seq(res_list):
    return ''.join([three_to_one.get(r,'X') for r in res_list])

def map_residues_to_indices(traj, residue_list):
    """Return CA atom indices corresponding to residue numbers in residue_list"""
    indices = []
    for res in traj.topology.residues:
        if res.is_protein and res.resSeq in residue_list:
            try:
                ca = res.atom(ATOM_NAME)
                indices.append(ca.index)
            except KeyError:
                continue
    return indices

# =========================
# Load trajectory
# =========================
print("Loading trajectory...")
traj = md.load_xtc(XTC_FILE, top=TOPOLOGY_PDB)

print("Loading references...")
active_ref = md.load_pdb(ACTIVE_PDB)
inactive_ref = md.load_pdb(INACTIVE_PDB)

# Map residues to CA indices
tm_idx   = map_residues_to_indices(traj, TM_RESIDUES)
pocket_idx = map_residues_to_indices(traj, BINDING_POCKET)

# =========================
# Alignment via PyMOL
# =========================
def pymol_align_traj_to_ref(traj, ref_pdb, tm_residues):
    """Align trajectory frames to reference using PyMOL iterative align"""
    aligned_coords = np.zeros_like(traj.xyz)
    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        # Load reference
        cmd.load(ref_pdb, "ref")
        # Loop over frames
        for i, frame in enumerate(traj):

            #temporary quick buffer
            if i % 100 != 0:
                continue

            tmp_pdb = f"tmp_frame_{i}.pdb"
            frame.save_pdb(tmp_pdb)
            cmd.load(tmp_pdb, "frame")
            # Align only TM residues by selection
            tm_sel = " or ".join([f"resi {r}" for r in tm_residues])
            cmd.align("frame and name CA and (" + tm_sel + ")", "ref and name CA and (" + tm_sel + ")")
            # Extract aligned coordinates
            cmd.create("aligned_frame", "frame")
            cmd.save(tmp_pdb, "aligned_frame")
            aligned_frame = md.load_pdb(tmp_pdb)
            aligned_coords[i] = aligned_frame.xyz[0]
            cmd.delete("frame")
            cmd.delete("aligned_frame")
            os.system("rm -drf tmp_frame_" + str(i) + ".pdb")
    return traj.__class__(xyz=aligned_coords, topology=traj.topology)

# Align trajectory to active and inactive references
print("Aligning trajectory to active state using PyMOL...")
traj_aligned_active = pymol_align_traj_to_ref(traj, ACTIVE_PDB, TM_RESIDUES)

print("Aligning trajectory to inactive state using PyMOL...")
traj_aligned_inactive = pymol_align_traj_to_ref(traj, INACTIVE_PDB, TM_RESIDUES)

# =========================
# Compute RMSD over binding pocket residues
# =========================
rmsd_active   = md.rmsd(traj_aligned_active, active_ref, atom_indices=pocket_idx)
rmsd_inactive = md.rmsd(traj_aligned_inactive, inactive_ref, atom_indices=pocket_idx)

# =========================
# Save results
# =========================
np.savetxt(
    "rmsd_pocket_per_state.dat",
    np.column_stack([rmsd_active, rmsd_inactive]),
    header="RMSD_active  RMSD_inactive"
)

print("Frame   RMSD_active   RMSD_inactive  ΔRMSD")
for i, (ra, ri) in enumerate(zip(rmsd_active, rmsd_inactive)):
    delta = ri - ra
    print(f"{i:5d}   {ra:12.4f}   {ri:14.4f}   {delta:10.4f}")

# =========================
# Save last frames for visual inspection
# =========================
traj_aligned_active[-1].save_pdb("last_frame_aligned_to_active.pdb")
traj_aligned_inactive[-1].save_pdb("last_frame_aligned_to_inactive.pdb")

print("\nDone. Last frames saved for visual inspection.")
