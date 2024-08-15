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
parser.add_argument(
    "--snap",
    type=str,
    help="The snapshot to plot.",
    default="010_z005p000",
)

args = parser.parse_args()

# Unpack snapshot
snap = args.snap

# Include units on the distance and convert to Mpc
dist = (args.dist * kpc).to(Mpc).value

# Define mass bins
mass_bins = [10**8, 10**9, 10**9.5, 10**10, np.inf]

# Loop over regions and snapshots calculating the pair distances
pair_dists = {}
with h5py.File(args.master_file, "r") as hdf:
    for reg in tqdm(hdf.keys(), desc="Regions"):
        for mass_low, mass_high in zip(mass_bins[:-1], mass_bins[1:]):
            # Create an entry for the mass bin
            pair_dists.setdefault(mass_low, [])

            # Extract the redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Get the galaxy group
            gal_grp = hdf[f"{reg}/{snap}/Galaxy"]

            # Get the positions and convert to physical units
            pos = to_physical(gal_grp["COP"][:].T, z)

            # Get the masses
            mass = gal_grp["Mstar_aperture/30"][:] * 10**10

            # Mask the masses for this mass bin
            mask = (mass > mass_low) & (mass < mass_high)

            # Create a KDTree
            tree = cKDTree(pos[mask, :])

            # Query the tree
            pairs = tree.query_pairs(dist, output_type="set")

            # Calculate the distances
            dists = (
                np.array([np.linalg.norm(pos[i] - pos[j]) for i, j in pairs])
                * 1000
            )

            # Store the distances
            pair_dists[mass_low].extend(dists)

# Plot the histogram for each snapshot
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.loglog()
ax.grid(True)
ax.set_axisbelow(True)

# Define the colors
colors = plt.cm.viridis(np.linspace(0, 1, len(pair_dists)))

# Create the bins from the minimum distance (across all mass bins) to
# the maixmum dist
min_dist = min([min(dists) for dists in pair_dists.values()])
bins = np.logspace(np.log10(min_dist), np.log10(dist * 1000), args.nbins)

# Loop over the mass bins and plot the histogram
for i, (mass_low, mass_high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
    # Plot the histogram
    ax.hist(
        pair_dists[mass_low],
        bins=bins,
        histtype="step",
        color=colors[i],
        label=f"${np.log10(mass_low):.0f} < "
        r"\log_{10}(M_\star/M_\odot) "
        f"< {np.log10(mass_high):.0f}$"
        if mass_high < np.inf
        else r"$\log_{10}(M_\star/M_\odot) " f"> {np.log10(mass_low):.0f}$",
    )

ax.set_xlabel("$R_{i,j} / $ [pkpc]")
ax.set_ylabel("$N$")

# Place legend below x axis
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

# Save the figure
savefig(fig, args.output_file + f"_{snap}")
