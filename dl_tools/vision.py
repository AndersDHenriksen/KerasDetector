import numpy as np
from PIL import Image
is_matplotlib_available = True
try:
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
except ImportError:
    is_matplotlib_available = False


def showimg(img, overlay_mask=None, close_on_click=False, cmap="gray", overlay_cmap="RdBu"):
    if not is_matplotlib_available:
        Image.fromarray(img).show()
        return

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5, forward=True)
    # Show image
    img = np.squeeze(img)
    if img.ndim == 1:
        plt.plot(img)
    else:
        plt.imshow(img, cmap=cmap)
        if overlay_mask is not None:
            masked = np.ma.masked_where(overlay_mask == 0, overlay_mask)
            plt.imshow(masked, overlay_cmap, alpha=0.5)
    # Trim margins
    plt.tight_layout()
    if close_on_click:
        plt.waitforbuttonpress()
        plt.close(fig)
    else:
        plt.show()
    return fig
