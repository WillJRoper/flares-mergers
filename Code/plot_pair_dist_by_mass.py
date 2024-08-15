"""A script for plotting a histogram of pair distances split by mass."""
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
    default="pair_dists_split_by_mass",
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

# Define mass bins
mass_bins = [10**8, 10**9, 10**9.5, 10**10, np.inf]

# Loop over regions and snapshots calculating the pair distances
for snap in PROG_SNAPS:
    # Get the prog snap
    prog_snap = PROG_SNAPS[snap]
    pair_dists = {}
    with h5py.File(args.master_file, "r") as hdf:
        for reg in tqdm(hdf.keys(), desc="Regions"):
            for mass_low, mass_high in zip(mass_bins[:-1], mass_bins[1:]):
                # Create an entry for the snapshot
                pair_dists.setdefault(mass_low, [])

                # Extract the redshift
                z = float(prog_snap.split("z")[-1].replace("p", "."))

                # Get the galaxy group
                gal_grp = hdf[f"{reg}/{snap}/Galaxy"]

                # Get the positions and convert to physical units
                pos = to_physical(gal_grp["COP"][:].T, z)

                # Get the masses
                mass = gal_grp["Mstar_aperture/30"][:] * 10**10

                # Mask the masses for this mass bin
                mask = (mass > mass_low) & (mass < mass_high)
                pos = pos[mask]

                # Create a KDTree
                tree = cKDTree(pos)

                # Query the tree
                pairs = tree.query_pairs(dist, output_type="set")

                # Calculate the distances
                dists = (
                    np.array(
                        [np.linalg.norm(pos[i] - pos[j]) for i, j in pairs]
                    )
                    * 1000
                )

                # Store the distances
                pair_dists[mass_low].extend(dists)

    # Loop over regions and snapshots calculating the distances between progenitors
    prog_pair_dists = {}
    with h5py.File(args.master_file, "r") as hdf:
        for reg in hdf.keys():
            for mass_low, mass_high in zip(mass_bins[:-1], mass_bins[1:]):
                # Get this snapshots progenitor data
                this_start_inds = np.array(start_inds[reg][snap], dtype=int)
                this_nprogs = np.array(nprogs[reg][snap], dtype=int)

                # Create an entry for the snapshot
                prog_pair_dists.setdefault(mass_low, [])

                # Extract the redshift
                z = float(prog_snap.split("z")[-1].replace("p", "."))

                # Get the galaxy group
                gal_grp = hdf[f"{reg}/{prog_snap}/Galaxy"]

                # Get the positions and convert to physical units
                pos = to_physical(gal_grp["COP"][:].T, z)

                # Get the masses
                mass = gal_grp["Mstar_aperture/30"][:] * 10**10

                # Get the distances for the progenitors
                dists = []
                for i in tqdm(
                    range(len(this_start_inds)),
                    desc=f"Progenitors for {reg}:{snap}",
                ):
                    inds = list(
                        range(
                            this_start_inds[i],
                            this_start_inds[i] + this_nprogs[i],
                        )
                    )

                    # Get the mass and positions for these progenitors
                    prog_mass = mass[inds]
                    prog_pos = pos[inds, :]

                    # Filter for galaxies with mass > 10^8 Msun
                    mask = mass > 10**8
                    prog_pos = prog_pos[mask]

                    if len(prog_mass[mask]) < 2:
                        continue

                    # Create a KDTree
                    tree = cKDTree(prog_pos)

                    # Query the tree
                    pairs = tree.query_pairs(dist, output_type="set")

                    # Calculate the distances
                    prog_dists = (
                        np.array(
                            [
                                np.linalg.norm(prog_pos[i] - prog_pos[j])
                                for i, j in pairs
                            ]
                        )
                        * 1000
                    )

                    # Store the distances
                    prog_pair_dists[mass_low].extend(prog_dists)

    # Plot the histogram for each snapshot in two panels (one above the other)
    fig = plt.figure(figsize=(3.5, 2 * 3.5))
    gs = fig.add_gridspec(
        2,
        1,
        hspace=0.0,
        width_ratios=[20, 1],
    )
    ax = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
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
    min_dist = min([min(pair_dists[mass_low]) for mass_low in pair_dists])
    bins = np.logspace(np.log10(min_dist), np.log10(dist * 1000), args.nbins)
    bin_cents = (bins[1:] + bins[:-1]) / 2

    # Loop over the mass bins
    for i, (mass_low, mass_high) in enumerate(
        zip(mass_bins[:-1], mass_bins[1:])
    ):
        # Plot the histogram
        n, _, _ = ax.hist(
            pair_dists[mass_low],
            bins=bins,
            histtype="step",
            color=colors[i],
            label=f"${np.log10(mass_low):.0f} < "
            r"\log_{10}(M_\star/M_\odot) "
            f"< {np.log10(mass_high):.0f}$"
            if mass_high < np.inf
            else r"$\log_{10}(M_\star/M_\odot) "
            f"> {np.log10(mass_low):.0f}$",
        )

        # Skip missing snaps
        if snap not in prog_pair_dists:
            continue

        # Plot the histogram for the progenitors
        prog_n, _, _ = ax1.hist(
            prog_pair_dists[mass_low],
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

    # Place legend below x axis
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Save the figure
    savefig(fig, args.output_file + f"_{snap}")
