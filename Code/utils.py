"""General utlity functions for the project."""
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np

# Define the snapshots list
SNAPSHOTS = [
    "000_z015p000",
    "001_z014p000",
    "002_z013p000",
    "003_z012p000",
    "004_z011p000",
    "005_z010p000",
    "006_z009p000",
    "007_z008p000",
    "008_z007p000",
    "009_z006p000",
    "010_z005p000",
]

PROG_SNAPS = {
    snap: prog_snap for snap, prog_snap in zip(SNAPSHOTS[1:], SNAPSHOTS[:-1])
}


def savefig(fig, outname):
    """
    Save a figure to the plots directory.

    Args:
        fig (Figure): The figure to save.
        outname (str): The name of the output file.
    """
    # Split the directory and file name
    dirname, basename = os.path.split(outname)

    # Create the directory if it doesn't exist
    if dirname:
        os.makedirs("../plots/" + dirname, exist_ok=True)

    # Save the figure
    fig.savefig(
        f"../plots/{outname}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def to_physical(pos, redshift):
    """
    Convert a comoving position to a physical position.

    Args:
        pos (array): The comoving position.
        redshift (float): The redshift of the snapshot.

    Returns:
        array: The physical position.
    """
    return pos / (1 + redshift)


def unpack_progenitors(filepath):
    """
    Read the progenitor information from the file and make it more accessible.

    Args:
        filepath (str): The path to the file.

    Returns:
        dict: The progenitor information.
    """
    # Create a dictionaries to store the progenitors
    start_indexes = {}
    nprogenitors = {}

    # Open the file
    with h5py.File(filepath, "r") as hdf:
        for reg in hdf.keys():
            for snap in hdf[reg].keys():
                # Make entries for this snapshot
                start_indexes.setdefault(snap, [])
                nprogenitors.setdefault(snap, [])

                # Get galaxy group
                gal_group = hdf[reg][snap]["Galaxy"]

                # Get mergergraph group
                merger_group = gal_group[reg][snap]["MergerGraph"]

                # Get the data
                starts = merger_group["Prog_Start_Index"]
                nprogs = merger_group["NProgs"]

                # Shift the start indexes
                starts = starts + np.sum(nprogenitors[snap])

                # Store the data
                start_indexes[snap].extend(starts)
                nprogenitors[snap].extend(nprogs)

    return start_indexes, nprogenitors
