from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as plticker


def plot_image_from_array(
    img_array,
    height=None,
    width=None,
    bounding_boxes=None,
    plot_grids=False,
    grid_size=None,
    color="b",
):
    """Creates a plot from a numpy array

    Parameters
    ----------
    img_array : numpy array
        Image to be plotted
    height : int, optional
        Height of the matplotlib.pyplot can be adjusted.
    width : int, optional
        Width of the matplotlib.pyplot can be adjusted.
    bounding_boxes : list, optional
        List of lists of bounding box coordinates: [[x1, y1, x2, y2]]
        with (x1, y1) being the coordinates of the top left corner
        and (x2, y2) the coordinates of the bottom right corner.
    plot_grids : bool, optional
        Indicates whether a grid should be plotted onto the image or not (default = Fales).
    grid_size : int, optional
        Size of each square in the grid to be plotted (use only if plot_grids = True).
    color :
        Color of text in each square in the grid (use only if plot_grids = True).

    Returns
    -------
    None
    """

    img = Image.fromarray((255 * img_array).astype("uint8")).convert("RGB")
    if height == None or width == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=(height, width))

    if plot_grids:
        loc = plticker.MultipleLocator(base=grid_size)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

        # Add the grid
        ax.grid(which="major", axis="both", linestyle="-")

    ax.imshow(img)

    if plot_grids:

        # Find number of gridsquares in x and y direction
        nx = abs(int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(grid_size)))
        ny = abs(int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(grid_size)))

        # Add some labels to the gridsquares
        for j in range(ny):

            y = grid_size / 2 + j * grid_size
            for i in range(nx):
                x = grid_size / 2.0 + float(i) * grid_size
                ax.text(x, y, "{:d}".format(i), color=color, ha="center", va="center")
                # ax.text(x, y, '{:d}'.format(i + j * nx), color=color, ha='center', va='center')

    # plot bounding boxes
    if type(bounding_boxes) != None:
        for bounding_box in bounding_boxes:
            x1, y1, x2, y2 = bounding_box
            bottom_left_corner = (x1, y2)
            box_width = x2 - x1
            box_height = y2 - y1
            rect = patches.Rectangle(
                bottom_left_corner,
                box_width,
                -box_height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()
