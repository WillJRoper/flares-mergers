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
    "--dist",
    type=float,
    help="The maximum pair distance in pkpc.",
    default=100,
)
parser.add_argument(
    "--nbins",
    type=int,
    help="The number of bins.",
    default=30,
)

args = parser.parse_args()

# Include units on the distance and convert to Mpc
dist = (args.dist * kpc).to(Mpc).value

# Loop over regions and snapshots calculating the pair distances
pair_dists = {}
with h5py.File(args.master_file, "r") as hdf:
    for reg in tqdm(hdf.keys(), desc="Regions"):
        for snap in hdf[reg].keys():
            # Create an entry for the snapshot
            pair_dists.setdefault(snap, [])

            # Extract the redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Get the galaxy group
            gal_grp = hdf[f"{reg}/{snap}/Galaxy"]

            # Get the positions and convert to physical units
            pos = to_physical(gal_grp["COP"][:].T, z)

            # Get the masses
            mass = gal_grp["Mstar_aperture/30"][:] * 10**10

            # Filter for galaxies with mass > 10^8 Msun
            mask = mass > 10**8
            pos = pos[mask]

            # Create a KDTree
            tree = cKDTree(pos)

            # Query the tree
            pairs = tree.query_pairs(dist, output_type="set")

            # Calculate the distances
            dists = (
                np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in pairs])
                * 1000
            )

            # Store the distances
            pair_dists[snap].extend(dists)

# Plot the histogram for each snapshot
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.loglog()
ax.grid(True)
ax.set_axisbelow(True)

# Create a colormap for each redshift
zs = np.arange(5, 16, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(zs)))

# Create the bins from the minimum distance (across all snapshots) to
# the maixmum dist
min_dist = min([min(pair_dists[snap]) for snap in pair_dists])
bins = np.logspace(np.log10(min_dist), np.log10(dist * 1000), args.nbins)

# Loop over the snapshots
for i, snap in enumerate(sorted(pair_dists.keys())):
    # Plot the histogram
    ax.hist(
        pair_dists[snap],
        bins=bins,
        histtype="step",
        color=colors[i],
    )

ax.set_xlabel("$R_{i,j} / $ [pkpc]")
ax.set_ylabel("$N$")

# Create the colorbar for the redshifts
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="viridis"),
    ax=ax,
    ticks=np.linspace(0, 1, len(zs)),
)
cbar.ax.set_yticklabels(reversed([f"{z:.0f}" for z in zs]))
cbar.set_label("$z$")
cbar.ax.invert_yaxis()

# Save the figure
savefig(fig, args.output_file)
