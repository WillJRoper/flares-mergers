"""General utlity functions for the project."""
import os
import matplotlib.pyplot as plt


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
        f"plots/{outname}.png",
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
