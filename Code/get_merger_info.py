"""Script for making convnient datasets for merger classification."""
import h5py
from scipy.spatial import cKDTree
import numpy as np

from pair_obj import GalaxyPair
from utils import to_physical


def make_pairs(filepath, reg, snap, d=0.050):
    """
    Get a list of pairs of galaxies.

    Args:
        filepath (str): The path to the hdf5 file.
        d (float): The distance threshold for a pair to be considered.

    Returns:
        list: A list of GalaxyPair objects.
    """
    # Open the file and collect all the data we'll need
    with h5py.File(filepath, "r") as hdf:
        # Get the galaxy group
        gal_group = hdf[reg][snap]["Galaxy"]

        # Get the merger group
        if "MergerGraph" in gal_group:
            merger_group = gal_group["MergerGraph"]

        # Get the galaxy centred of potentials
        gal_pos = to_physical(
            gal_group["COP"][:].T, float(snap.split("z")[-1].replace("p", "."))
        )

        # Get Group Numbers
        group_nums = gal_group["GroupNumber"][:]
        subgroup_nums = gal_group["SubGroupNumber"][:]

        # Get the masses
        masses = gal_group["Mstar_aperture/30"][:] * 10**10

        # Get the merger data
        if "MergerGraph" in gal_group:
            desc_groups = merger_group["desc_group_ids"][:]
            desc_subgroups = merger_group["desc_subgroup_ids"][:]
            desc_pointers = merger_group["Desc_Start_Index"][:]
            ndescs = merger_group["nDescs"][:]
        else:
            desc_groups = np.zeros(0)
            desc_subgroups = np.zeros(0)
            desc_pointers = np.zeros(len(masses), dtype=int)
            ndescs = np.zeros(len(masses), dtype=int)

    # Filter for "resolved" galaxies
    mask = masses > 10**8
    gal_pos = gal_pos[mask]
    group_nums = group_nums[mask]
    subgroup_nums = subgroup_nums[mask]
    masses = masses[mask]

    # Create a KDTree
    tree = cKDTree(gal_pos)

    # Get all the pairs
    pairs = tree.query_pairs(d, output_type="set")

    # Loop over the pairs and create the pair objects
    pair_objs = []
    for i, j in pairs:
        # Calculate the distance between the galaxies
        dist = np.linalg.norm(gal_pos[i] - gal_pos[j])

        # Get the descendents for each galaxy
        desc_grps_i = desc_groups[
            desc_pointers[i] : desc_pointers[i] + ndescs[i]
        ]
        desc_subgrps_i = desc_subgroups[
            desc_pointers[i] : desc_pointers[i] + ndescs[i]
        ]
        desc_grps_j = desc_groups[
            desc_pointers[j] : desc_pointers[j] + ndescs[j]
        ]
        desc_subgrps_j = desc_subgroups[
            desc_pointers[j] : desc_pointers[j] + ndescs[j]
        ]
        desc_ids_i = set(zip(desc_grps_i, desc_subgrps_i))
        desc_ids_j = set(zip(desc_grps_j, desc_subgrps_j))

        # Do they share a common descendent?
        if len(desc_ids_i) == 0 or len(desc_ids_j) == 0:
            is_merger = False
        elif len(desc_ids_i) > 0 and len(desc_ids_j) > 0:
            is_merger = bool(desc_ids_i & desc_ids_j)
        else:
            is_merger = False

        pair_objs.append(
            GalaxyPair(
                group_nums[i],
                subgroup_nums[i],
                dist,
                snap,
                masses[i],
                masses[j],
                is_merger,
            )
        )

    return pair_objs
