import mdtraj as md



traj = md.load_xtc("samples_sidechain_rec.xtc", top="samples_sidechain_rec.pdb")

print(traj.n_frames)


