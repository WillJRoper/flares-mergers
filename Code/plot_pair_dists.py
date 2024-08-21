"""A script for plotting a histogram of pair distances."""
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from unyt import kpc, Mpc

from utils import savefig, unpack_progenitors, SNAPSHOTS, REGIONS
from get_merger_info import make_pairs


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

# Get progenitor information
start_inds, nprogs = unpack_progenitors(args.master_file)

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
fig = plt.figure(figsize=(3.5 * 1.1, 2 * 3.5))
gs = fig.add_gridspec(
    4,
    2,
    hspace=0.0,
    width_ratios=[20, 1],
)
ax = fig.add_subplot(gs[0:2, 0])
ax1 = fig.add_subplot(gs[2:, 0])
cax = fig.add_subplot(gs[1:3, 1])
ax.grid(True)
ax1.grid(True)
ax.set_axisbelow(True)
ax1.set_axisbelow(True)
ax.set_xscale("log")
ax1.set_xscale("log")
ax.set_yscale("log")
ax1.set_yscale("log")

# Remove the xticks from the upper plot
ax.set_xticklabels([])

# Create a colormap for each redshift
zs = np.arange(5, 16, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(zs)))

# Create the bins from the minimum distance (across all snapshots) to
# the maixmum dist
min_dist = min([min(pair_dists[snap]) for snap in pair_dists])
bins = np.logspace(np.log10(min_dist), np.log10(dist * 1000), args.nbins)
bin_cents = (bins[1:] + bins[:-1]) / 2

# Loop over the snapshots
for i, snap in enumerate(sorted(pair_dists.keys())):
    # Plot the histogram
    n, _, _ = ax.hist(
        pair_dists[snap],
        bins=bins,
        histtype="step",
        color=colors[i],
    )

    # Plot the histogram for the progenitors
    prog_n, _, _ = ax1.hist(
        merger_dists[snap],
        bins=bins,
        histtype="step",
        color=colors[i],
    )

# Put a text box in the top left corner
ax.text(
    0.05,
    0.95,
    "All Galaxies",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.5"),
)
ax1.text(
    0.05,
    0.95,
    "Mergers",
    transform=ax1.transAxes,
    fontsize=8,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.5"),
)

ax1.set_xlabel("$R_{i,j} / $ [pkpc]")
ax.set_ylabel("$N$")
ax1.set_ylabel("$N$")

# Set y lims to match
ax1.set_ylim(ax.get_ylim())

# Create the colorbar for the redshifts
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="viridis"),
    cax=cax,
    ticks=np.linspace(0, 1, len(zs)),
)
cbar.ax.set_yticklabels(reversed([f"{z:.0f}" for z in zs]))
cbar.set_label("$z$")
cbar.ax.invert_yaxis()

# Save the figure
savefig(fig, args.output_file)
