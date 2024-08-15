"""A script for plotting a histogram of pair distances."""
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from unyt import kpc, Mpc
from scipy.spatial import cKDTree

from utils import savefig, to_physical


# Define the inputs
parser = argparse.ArgumentParser()
parser.add_argument(
    "--master_file",
    type=str,
    help="The path to the master file.",
    default="../data/flares_with_mergers.hdf5",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="The path to the output file.",
    default="pair_dists",
)
parser.add_argument(
    "--dist", type=float, help="The maximum pair distance in pkpc.", default=50
)

args = parser.parse_args()

# Include units on the distance and convert to Mpc
dist = (args.dist * kpc).to(Mpc).value

# Loop over regions and snapshots calculating the pair distances
pair_dists = {}
with h5py.File(args.master_file, "r") as hdf:
    for reg in tqdm(hdf.keys(), desc="Regions"):
        for snap in tqdm(hdf[reg].keys(), desc="Snapshots", leave=False):
            # Create an entry for the snapshot
            pair_dists.setdefault(snap, [])

            # Extract the redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Get the galaxy group
            gal_grp = hdf[f"{reg}/{snap}/Galaxy"]

            # Get the positions and convert to physical units
            pos = to_physical(gal_grp["COP"][:], z)

            # Create a KDTree
            tree = cKDTree(pos)

            # Query the tree
            dists, _ = tree.query_pairs(dist, r=dist, routput_type="ndarray")

            # Store the distances
            pair_dists[snap].extend(dists)

# Plot the histogram for each snapshot
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.loglog()
ax.grid(True)
ax.setaxisbelow(True)

# Create a colormap for each redshift
zs = np.arange(5, 16, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(zs)))

# Loop over the snapshots
for i, snap in enumerate(sorted(pair_dists.keys())):
    # Plot the histogram
    ax.hist(
        pair_dists[snap],
        bins=np.logspace(0, np.log10(dist), 50),
        histtype="step",
        color=colors[i],
    )

ax.set_xlabel("$R_{i,j} / $ [Mpc]")
ax.set_ylabel("$N$")

# Create the colorbar
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax)
cbar.set_label("$z$")
cbar.set_ticks(np.linspace(0, 1, len(zs)))
cbar.set_ticklabels(zs)

# Save the figure
savefig(fig, args.output_file)
