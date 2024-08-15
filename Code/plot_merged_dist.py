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

# Get progenitor information
start_inds, nprogs = unpack_progenitors(args.master_file)

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

# Loop over regions and snapshots calculating the distances between progenitors
prog_pair_dists = {}
with h5py.File(args.master_file, "r") as hdf:
    for reg in hdf.keys():
        for snap in hdf[reg].keys():
            # Can't do the first snapshot since there are no progenitors
            if snap not in PROG_SNAPS:
                continue

            # Get the prog snap
            prog_snap = PROG_SNAPS[snap]

            # Get this snapshots progenitor data
            this_start_inds = np.array(start_inds[reg][snap], dtype=int)
            this_nprogs = np.array(nprogs[reg][snap], dtype=int)

            # Create an entry for the snapshot
            prog_pair_dists.setdefault(prog_snap, [])

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
                        this_start_inds[i], this_start_inds[i] + this_nprogs[i]
                    )
                )

                # Get the mass and positions for these progenitors
                prog_mass = mass[inds]
                prog_pos = pos[inds, :]

                # Mask out progentiros below 10^8 Msun
                mask = prog_mass > 10**8
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
                prog_pair_dists[prog_snap].extend(prog_dists)


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
    # Can't dp the final snapshot because it has no progenitors
    if "z005p000" in snap:
        continue

    # Extract the redshift
    z = float(snap.split("z")[-1].replace("p", "."))
    zs.append(z)

    # Loop over distance thresh holds
    for d in dist_threshes:
        # Get the number of pairs below the threshold
        n_prog = np.sum(np.array(prog_pair_dists[snap]) < d)

        # If we have no mergers store a 0
        if n_prog == 0:
            fracs.setdefault(d, []).append(0)
            continue

        # Get the total number of pairs
        n = np.sum(np.array(pair_dists[snap]) < d)

        # Store the fraction
        fracs.setdefault(d, []).append(n_prog / n)

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
ax.set_ylabel("$N$")

# Create the colorbar for the distance thresholds
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="viridis"),
    cax=cax,
)
cbar.set_ticks(np.linspace(0, 1, len(dist_threshes)))
cbar.set_ticklabels([f"{d:.1f}" for d in dist_threshes])
cbar.set_label("$R_{i,j} < d / [pkpc]$")

# Save the figure
savefig(fig, args.output_file)
