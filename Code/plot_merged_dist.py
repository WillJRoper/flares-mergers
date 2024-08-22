"""A script for plotting the fractions of pairs which have merged."""
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from unyt import kpc, Mpc
from scipy.spatial import cKDTree

from utils import savefig, to_physical, unpack_progenitors, PROG_SNAPS


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
    default="fraction_merged_at_dist",
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

# Include units on the distance and convert to Mpc
dist = (args.dist * kpc).to(Mpc).value

# Loop over regions and snapshots calculating the pair distances
pair_dists = {}
merger_dists = {}
for reg in tqdm(REGIONS, desc="Regions"):
    for snap in SNAPSHOTS[:-1]:
        # Get the galaxy pairs
        galaxy_pairs = make_pairs(args.master_file, reg, snap, d=dist)

        # Extract the distances
        for pair in galaxy_pairs:
            pair_dists.setdefault(snap, []).append(pair.dist)
            if pair.is_merger:
                merger_dists.setdefault(snap, []).append(pair.dist)


# Plot the histogram for each snapshot in two panels (one above the other)
fig = plt.figure(figsize=(3.5 * 1.1, 3.5))
gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[20, 1],
)
ax = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])
ax.grid(True)
ax.set_axisbelow(True)

# Define distance thresholds
dist_threshes = [1, 5, 10, 50, 100]

# Define the colors
colors = plt.cm.viridis(np.linspace(0, 1, len(dist_threshes)))

# Loop over the snapshots and get the data
zs = []
fracs = {}
for i, snap in enumerate(sorted(pair_dists.keys())):
    # Extract the redshift
    z = float(snap.split("z")[-1].replace("p", "."))
    zs.append(z)

    # Loop over distance thresh holds
    for d in dist_threshes:
        # Get the number of pairs below the threshold
        n_merged = np.sum(np.array(merger_dists[snap]) < d)

        # If we have no mergers store a 0
        if n_merged == 0:
            fracs.setdefault(d, []).append(0)
            continue

        # Get the total number of pairs
        n = np.sum(np.array(pair_dists[snap]) < d)

        # Store the fraction
        fracs.setdefault(d, []).append(n_merged / n)

# Convert to arrays
zs = np.array(zs)
for d in dist_threshes:
    fracs[d] = np.array(fracs[d])

# Swap zeros for nans
for d in dist_threshes:
    fracs[d][fracs[d] == 0] = np.nan

# Plot each distance
for i, d in enumerate(dist_threshes):
    ax.plot(
        zs,
        fracs[d],
        color=colors[i],
    )

ax.set_xlabel("$z$")
ax.set_ylabel("$N_{\mathrm{merg}}(< d) / N_{\mathrm{all}}(< d)$")

# Create the colorbar for the distance thresholds
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="viridis"),
    cax=cax,
)
cbar.set_ticks(np.linspace(0, 1, len(dist_threshes)))
cbar.set_ticklabels([f"{d:.1f}" for d in dist_threshes])
cbar.set_label("$d / [pkpc]$")

# Save the figure
savefig(fig, args.output_file)
