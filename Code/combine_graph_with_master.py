"""Combine the FLARES master file and mergergraph file."""
import h5py
import numpy as np


def extract_group_subgroup_ids(float_id):
    """
    Extracts the group ID and subgroup ID from a combined float representation.

    Parameters:
    float_id (float): The combined float of the form grpID.%05d' %subgrpID.

    Returns:
    tuple: A tuple containing the group ID (int) and subgroup ID (int).
    """
    # Separate the integer part (grpID) and the fractional part
    grpID = int(float_id)
    fractional_part = float_id - grpID

    # Multiply the fractional part by 100000 and round it to get subgrpID
    subgrpID = int(fractional_part * 100000)

    return grpID, subgrpID


def exclude_and_copy_group(src_group, dest_group, exclude_group_names):
    """
    Recursively copies contents from src_group to dest_group, excluding the specified group.
    """
    for item_name, item in src_group.items():
        # Construct the path for the current item
        path = f"{src_group.name}/{item_name}".lstrip("/")

        # Skip the excluded group and its contents
        skip = False
        for exclude_group_name in exclude_group_names:
            if exclude_group_name in path:
                skip = True
                break
        if skip:
            continue

        print(f"Copying {path}")

        # If it's a group, create it in the destination and recurse
        if isinstance(item, h5py.Group):
            dest_sub_group = dest_group.require_group(item_name)
            exclude_and_copy_group(item, dest_sub_group, exclude_group_names)
        # If it's a dataset, copy it directly
        elif isinstance(item, h5py.Dataset):
            src_group.copy(item_name, dest_group)


def copy_hdf5_excluding_group(
    original_file_path, new_file_path, exclude_group_names
):
    with h5py.File(original_file_path, "r") as original_file, h5py.File(
        new_file_path, "w"
    ) as new_file:
        # Start the recursive copying from the root, excluding the specified group
        exclude_and_copy_group(original_file, new_file, exclude_group_names)


# Define the input files
master_file = (
    "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"
)
new_file = "flares_with_mergers.hdf5"

# Define the mergergraph directory
mega_path = "/cosma7/data/dp004/FLARES/FLARES-1/MergerGraphs/"

# Make a new copy
copy_hdf5_excluding_group(
    master_file,
    new_file,
    ["Particle", "BPASS"],
)

# Define the regions
regs = []
for reg in range(40):
    regs.append(str(reg).zfill(2))

# Define  the snapshots
flares_snaps = [
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
    "011_z004p770",
]


with h5py.File(new_file, "r+") as hdf_master:
    # Loop over the regions
    for reg in regs:
        # Loop over the snapshots
        for snap in flares_snaps:
            print(reg, snap)
            # Get the galaxy group
            gal_grp = hdf_master[f"{reg}/{snap}/Galaxy"]

            # Define the mergergraph file
            mergergraph_file = (
                mega_path + f"GEAGLE_{reg}/SubMgraph_{snap}.hdf5"
            )
            # Open the mergergraph file
            with h5py.File(mergergraph_file, "r") as hdf_mergergraph:
                # Get the master group and subgroup IDs
                master_grpIDs = gal_grp["GroupNumber"][:]
                master_subgrpIDs = gal_grp["SubGroupNumber"][:]

                # Get the mergergraph group and subgroup IDs
                mergergraph_subfindIDs = hdf_mergergraph["SUBFIND_halo_IDs"][:]
                mergergraph_grpIDs = []
                mergergraph_subgrpIDs = []
                for subfindID in mergergraph_subfindIDs:
                    grpID, subgrpID = extract_group_subgroup_ids(subfindID)
                    mergergraph_grpIDs.append(grpID)
                    mergergraph_subgrpIDs.append(subgrpID)

                # Get the progenitor and descendant pointers and numbers
                prog_ptrs = hdf_mergergraph["Prog_Start_Index"][:]
                n_progs = hdf_mergergraph["nProgs"][:]
                desc_ptrs = hdf_mergergraph["Desc_Start_Index"][:]
                n_descs = hdf_mergergraph["nDescs"][:]

                # Make a nice look up table for the merger graph pointer and
                # length
                mergergraph_lookup = {}
                mergergraph_lookup_desc = {}
                for i, (grpID, subgrpID) in enumerate(
                    zip(mergergraph_grpIDs, mergergraph_subgrpIDs)
                ):
                    mergergraph_lookup[(grpID, subgrpID)] = (
                        prog_ptrs[i],
                        n_progs[i],
                    )
                    mergergraph_lookup_desc[(grpID, subgrpID)] = (
                        desc_ptrs[i],
                        n_descs[i],
                    )

                # Loop over galaxies in the master file getting the data
                pointers = np.zeros(len(master_grpIDs), dtype=np.int32)
                nprogs = np.zeros(len(master_grpIDs), dtype=np.int32)
                desc_ptrs = np.zeros(len(master_grpIDs), dtype=np.int32)
                ndescs = np.zeros(len(master_grpIDs), dtype=np.int32)
                prog_star_ms = []
                prog_grps = []
                prog_subgrps = []
                desc_star_ms = []
                desc_grps = []
                desc_subgrps = []
                for ind, (grp, subgrp) in enumerate(
                    zip(master_grpIDs, master_subgrpIDs)
                ):
                    # Skip galaxies without an entry
                    if (grp, subgrp) not in mergergraph_lookup:
                        nprogs[ind] = 0
                        pointers[ind] = -1
                        continue

                    # Get the mergergraph pointer and length
                    prog_start_index, nprog = mergergraph_lookup[(grp, subgrp)]
                    desc_start_index, ndesc = mergergraph_lookup_desc[
                        (grp, subgrp)
                    ]

                    # Get the progenitor masses
                    try:
                        prog_masses = hdf_mergergraph["prog_stellar_masses"][
                            prog_start_index : prog_start_index + nprog
                        ]
                        prog_grp = hdf_mergergraph["prog_group_ids"][
                            prog_start_index : prog_start_index + nprog
                        ]
                        prog_subgrp = hdf_mergergraph["prog_subgroup_ids"][
                            prog_start_index : prog_start_index + nprog
                        ]
                        desc_masses = hdf_mergergraph["desc_stellar_masses"][
                            desc_start_index : desc_start_index + ndesc
                        ]
                        desc_grp = hdf_mergergraph["desc_group_ids"][
                            desc_start_index : desc_start_index + ndesc
                        ]
                        desc_subgrp = hdf_mergergraph["desc_subgroup_ids"][
                            desc_start_index : desc_start_index + ndesc
                        ]
                    except KeyError:
                        continue

                    # Perform a stellar mass cut
                    okinds = prog_masses > 1e8 / 10**10
                    prog_masses = prog_masses[okinds]
                    prog_grp = prog_grp[okinds]
                    prog_subgrp = prog_subgrp[okinds]
                    okinds = desc_masses > 1e8 / 10**10
                    desc_masses = desc_masses[okinds]
                    desc_grp = desc_grp[okinds]
                    desc_subgrp = desc_subgrp[okinds]

                    # Sort by stellar mass (they're sorted by DM mass in the
                    # MEGA file)
                    if len(prog_masses) > 0:
                        sinds = np.argsort(prog_masses)
                        prog_masses = prog_masses[sinds[::-1]]
                        prog_grp = prog_grp[sinds[::-1]]
                        prog_subgrp = prog_subgrp[sinds[::-1]]
                    if len(desc_masses) > 0:
                        sinds = np.argsort(desc_masses)
                        desc_masses = desc_masses[sinds[::-1]]
                        desc_grp = desc_grp[sinds[::-1]]
                        desc_subgrp = desc_subgrp[sinds[::-1]]

                    # Store this data
                    nprogs[ind] = prog_masses.size
                    pointers[ind] = len(prog_star_ms)
                    ndescs[ind] = desc_masses.size
                    desc_ptrs[ind] = len(desc_star_ms)
                    prog_star_ms.extend(prog_masses)
                    desc_star_ms.extend(desc_masses)
                    prog_grps.extend(prog_grp)
                    prog_subgrps.extend(prog_subgrp)
                    desc_grps.extend(desc_grp)
                    desc_subgrps.extend(desc_subgrp)

                # Add the data to the master file under a new group
                gal_grp.create_group("MergerGraph")
                gal_grp["MergerGraph"].create_dataset(
                    "Prog_Start_Index", data=pointers
                )
                gal_grp["MergerGraph"].create_dataset("nProgs", data=nprogs)
                gal_grp["MergerGraph"].create_dataset(
                    "prog_stellar_masses", data=np.array(prog_star_ms)
                )
                gal_grp["MergerGraph"].create_dataset(
                    "prog_group_ids", data=np.array(prog_grps)
                )
                gal_grp["MergerGraph"].create_dataset(
                    "prog_subgroup_ids", data=np.array(prog_subgrps)
                )
                gal_grp["MergerGraph"].create_dataset(
                    "Desc_Start_Index", data=desc_ptrs
                )
                gal_grp["MergerGraph"].create_dataset("nDescs", data=ndescs)
                gal_grp["MergerGraph"].create_dataset(
                    "desc_stellar_masses", data=np.array(desc_star_ms)
                )
                gal_grp["MergerGraph"].create_dataset(
                    "desc_group_ids", data=np.array(desc_grps)
                )
                gal_grp["MergerGraph"].create_dataset(
                    "desc_subgroup_ids", data=np.array(desc_subgrps)
                )


def get_prog_data(ind, pointers, nprogs, prog_data):
    """
    Get the progenitor data for a galaxy.

    Args:
        ind (int):
            The index of the galaxy.
        pointers (np.ndarray):
            The pointers to first element of the progenitor data for each galaxy.
        nprogs (np.ndarray):
            The number of progenitors for each galaxy.
        prog_data (np.ndarray):
            The progenitor data to access.

    Returns:
        np.ndarray:
            The progenitor data for the galaxy.
    """
    # Get the start index and number of progenitors
    start_index = pointers[ind]
    nprog = nprogs[ind]

    # Get the progenitor data
    return prog_data[start_index : start_index + nprog]
