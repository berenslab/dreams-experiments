import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.widgets import Slider


def plot_embed(emb, eval, c, s=1.0, alpha=0.5, title=None, title_size=11, ax=None, show_eval=True, save_path=None, add_scale_plot=False, show_KNC = True, eval_size=10, white_edge=True, rasterized=False):
    """
    Plots an embedding with evaluation metrics.

    Parameters:
    - emb: np.ndarray or torch.Tensor
        The 2D embedding to plot, shape (N, 2).
    - eval: list or tuple
        Evaluation metrics, expected as (KNN, KNC, CPD).
    - c: np.ndarray
        Colors for each point in the embedding (either numerically->cmap or direct color names), shape (N,).
    - title: str, optional
        Title of the plot.
    - title_size: int/float, optional
        Font size of the title.
    - ax: matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    - show_eval: bool, optional
        Whether to show the evaluation metrics in plot.
    - save_path: str, optional
        Path to save the plot. If None, the plot is shown without saving plot.
    - add_scale_plot: bool, optional
        Whether to add a scale bar to the plot.
    - show_KNC: bool, optional
        Whether to show KNC in the evaluation metrics.
    - eval_size: int/float, optional
        Font size of the evaluation metrics.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Plot embedding
    scatter = ax.scatter(*emb.T, c=c, alpha=alpha, s=s, cmap="tab10", edgecolor="none", rasterized=rasterized)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_size)
        
    # Add evaluation text
    if show_eval:
        if show_KNC:
            text_eval = f"KNN: {eval[0]:.2f}\nKNC: {eval[1]:.2f}\nCPD: {eval[2]:.2f}"
        else:
            text_eval = f"KNN: {eval[0]:.2f}\nCPD: {eval[2]:.2f}"
        
        text = ax.text(
            1, 0, text_eval, ha='right', va='bottom', transform=ax.transAxes, fontsize=eval_size
        )
        if white_edge:
            text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),  
            path_effects.Normal()  
        ])

    # Add scale bar
    if add_scale_plot:
        add_scale(ax, emb)

    # Save or show plot
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if ax is None:
        plt.show()

def get_scale(embd, max_length=0.5):
    # returns the smallest power of 10 that is smaller than max_length * the
    # spread in the x direction
    spreads = embd.max(0) - embd.min(0)
    spread = spreads.max()
 
    return 10 ** (np.floor(np.log10(spread * max_length)))
 
def add_scale(ax, embd):
    """
    Adds a scale bar
    """
    scale = get_scale(embd)
    if embd.shape[1] == 2:
        embd_w, embd_h = embd.max(0) - embd.min(0)
        height = 0.005 * embd_h
 
    elif embd.shape[1] == 3:
        embd_w, embd_h, embd_d = embd.max(0) - embd.min(0)
        height = 0.00001 * embd_h
    else:
        raise NotImplementedError("Only 2D and 3D embeddings are supported")
 
    scalebar = AnchoredSizeBar(ax.transData,
                               scale,
                               str(scale),
                               loc="lower left",
                               size_vertical=height,
                               borderpad= 0.5,
                               sep=4,
                               frameon=False,
                               fontproperties={"size": 8})
    ax.add_artist(scalebar)
    return scalebar

def plot_embed_score(emb, eval, c, s=1.0, alpha=0.5, title=None, title_size=11, ax=None, show_eval=True, save_path=None, add_scale_plot=False, eval_size=10, white_edge=True, rasterized=False):
    """
    Plots an embedding with evaluation metrics.

    Parameters:
    - emb: np.ndarray or torch.Tensor
        The 2D embedding to plot, shape (N, 2).
    - eval: list or tuple
        Evaluation metrics, expected as (KNN, CPD, score).
    - c: np.ndarray
        Colors for each point in the embedding (either numerically->cmap or direct color names), shape (N,).
    - title: str, optional
        Title of the plot.
    - title_size: int/float, optional
        Font size of the title.
    - ax: matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    - show_eval: bool, optional
        Whether to show the evaluation metrics in plot.
    - save_path: str, optional
        Path to save the plot. If None, the plot is shown without saving plot.
    - add_scale_plot: bool, optional
        Whether to add a scale bar to the plot.
    - show_KNC: bool, optional
        Whether to show KNC in the evaluation metrics.
    - eval_size: int/float, optional
        Font size of the evaluation metrics.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Plot embedding
    scatter = ax.scatter(*emb.T, c=c, alpha=alpha, s=s, cmap="tab10", edgecolor="none", rasterized=rasterized)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_size)
        
    # Add evaluation text
    if show_eval:
        text_eval = f"KNN: {eval[0]:.2f}\nCPD: {eval[1]:.2f}\ns: {eval[2]:.2f}"
        
        text = ax.text(
            1, 0, text_eval, ha='right', va='bottom', transform=ax.transAxes, fontsize=eval_size
        )
        if white_edge:
            text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),  
            path_effects.Normal()  
        ])

    # Add scale bar
    if add_scale_plot:
        add_scale(ax, emb)

    # Save or show plot
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if ax is None:
        plt.show()
