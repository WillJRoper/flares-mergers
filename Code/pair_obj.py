"""A module defining a class for representing pairs of galaxies."""

from utils import SNAPSHOTS


class GalaxyPair:
    """
    A class for representing pairs of galaxies.

    Attributes:
        galaxy1 (Galaxy):
            The first galaxy in the pair.
        galaxy2 (Galaxy):
            The second galaxy in the pair.
        dist (float):
            The distance between the galaxies.
        mass1 (float):
            The mass of the first galaxy.
        mass2 (float):
            The mass of the second galaxy.
        snap_num (int):
            The snapshot number.
        snap (str):
            The snapshot name.
        desc_snap (str):
            The name of the snapshot after the current one.
        merger_ratio (float):
            The ratio of the masses of the galaxies.
    """

    def __init__(self, galaxy1, galaxy2, dist, snap, mass1, mass2, is_merger):
        """
        Initialize a GalaxyPair object.

        Args:
            galaxy1 (Galaxy):
                The first galaxy in the pair.
            galaxy2 (Galaxy):
                The second galaxy in the pair.
        """
        # Store the ids
        self.galaxy1 = galaxy1
        self.galaxy2 = galaxy2

        # The distance between the galaxies
        self.dist = dist

        # The masses of the galaxies
        self.mass1 = mass1
        self.mass2 = mass2

        # The snapshot number
        self.snap_num = int(snap.split("_")[0])
        self.snap = snap
        self.desc_snap = (
            SNAPSHOTS[self.snap_num + 1]
            if self.snap_num + 1 < len(SNAPSHOTS)
            else None
        )

        # Is this pair a merger?
        self.is_merger = is_merger

        # The merger ratio
        self.merger_ratio = min(self.mass1, self.mass2) / max(
            self.mass1, self.mass2
        )

    def is_major(self, thresh=0.25):
        """
        Determine if the pair is a major merger.

        Args:
            thresh (float):
                The threshold for the merger ratio.

        Returns:
            bool:
                True if the pair is a major merger, False otherwise.
        """
        return self.merger_ratio >= thresh

    def is_minor(self, thresh=0.25):
        """
        Determine if the pair is a minor merger.

        Args:
            thresh (float):
                The threshold for the merger ratio.

        Returns:
            bool:
                True if the pair is a minor merger, False otherwise.
        """
        return not self.is_major(thresh)

    def within_dist(self, dist_thresh=30):
        """
        Determine if the pair is within a given distance.

        Args:
            dist_thresh (float):
                The distance threshold.

        Returns:
            bool:
                True if the pair is within the distance threshold,
                False otherwise.
        """
        return self.dist <= dist_thresh
