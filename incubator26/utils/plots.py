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
    number_of_grids=None,
    color="b",
):

    img = Image.fromarray((255 * img_array).astype("uint8")).convert("RGB")
    if height == None or width == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=(height, width))

    if plot_grids:
        loc = plticker.MultipleLocator(base=number_of_grids)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

        # Add the grid
        ax.grid(which="major", axis="both", linestyle="-")

    ax.imshow(img)

    if plot_grids:

        # Find number of gridsquares in x and y direction
        nx = abs(
            int(float(ax.get_xlim()[1] - ax.get_xlim()[0]) / float(number_of_grids))
        )
        ny = abs(
            int(float(ax.get_ylim()[1] - ax.get_ylim()[0]) / float(number_of_grids))
        )

        # Add some labels to the gridsquares
        for j in range(ny):

            y = number_of_grids / 2 + j * number_of_grids
            for i in range(nx):
                x = number_of_grids / 2.0 + float(i) * number_of_grids
                ax.text(x, y, "{:d}".format(i), color=color, ha="center", va="center")
                # ax.text(x, y, '{:d}'.format(i + j * nx), color=color, ha='center', va='center')

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
