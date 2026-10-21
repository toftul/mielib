"""The paper figure conventions as functions.  Import, or copy what you need.

    from figure_style import use_paper_style, figure, letter, signed_map, inset_colorbar, save_raw

    use_paper_style(project_dir)                 # finds or creates phys-plots.mplstyle
    fig, axes = figure('double', rows=2, cols=3) # 6.2 in wide; 'single' is 3.2 in
    mesh = signed_map(axes[0, 0], x, y, F, label_zero=True)
    inset_colorbar(axes[1, 2], mesh, title=r'Radial force, $F_\rho/F_0$')
    letter(axes[0, 0], 'A')
    save_raw(fig, 'results', 'my_figure')        # results/my_figure_raw.{png,pdf}
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE_NAME = 'phys-plots.mplstyle'
WIDTHS = {'single': 3.2, 'double': 6.2}         # inches, one and two journal columns
ASPECT = 0.78                                    # default height/width of one panel


def find_style(project_dir='.'):
    """The project's style file, looked for upward from ``project_dir``.

    If none is found, the copy bundled with this skill is written into
    ``project_dir`` so the project owns it from then on.
    """
    here = Path(project_dir).resolve()
    for folder in [here, *here.parents][:4]:
        candidate = folder/STYLE_NAME
        if candidate.exists():
            return candidate
    bundled = [Path(__file__).resolve().parent/STYLE_NAME,
               Path.home()/'.claude'/'skills'/'sk-fig'/STYLE_NAME]
    for source in bundled:
        if source.exists():
            target = here/STYLE_NAME
            shutil.copy(source, target)
            print(f'no {STYLE_NAME} in the project: created {target}')
            return target
    return None


def use_paper_style(project_dir='.'):
    """Apply the style, then the settings an older style file may lack."""
    style = find_style(project_dir)
    if style is not None:
        plt.style.use(str(style))
    plt.rcParams['pdf.fonttype'] = 42            # text stays text in the PDF
    for key in ('axes.titlesize', 'figure.titlesize', 'legend.fontsize',
                'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize'):
        plt.rcParams[key] = plt.rcParams['font.size']   # one size for every font
    plt.rcParams['mathtext.fontset'] = 'custom'  # and one face: maths in the text font
    plt.rcParams['mathtext.rm'] = 'sans'
    plt.rcParams['mathtext.it'] = 'sans:italic'
    plt.rcParams['mathtext.bf'] = 'sans:bold'
    return style


def figure(width='single', rows=1, cols=1, height=None, **subplots_kw):
    """``plt.subplots`` at a journal width, constrained layout.

    ``width`` is ``'single'`` (3.2 in), ``'double'`` (6.2 in) or a number.
    ``height`` defaults to a panel aspect of ``ASPECT`` per row.
    """
    w = WIDTHS.get(width, width)
    h = height if height is not None else ASPECT*w/cols*rows
    subplots_kw.setdefault('layout', 'constrained')
    return plt.subplots(rows, cols, figsize=(w, h), **subplots_kw)


def letter(ax, text, x=0.0, y=1.03):
    """Bold panel letter outside the axes, top left."""
    ax.text(x, y, text, transform=ax.transAxes, fontweight='bold', ha='left', va='bottom')


def edges(centers):
    """Cell edges for ``pcolormesh(shading='flat')`` from cell centers."""
    c = np.asarray(centers)
    return np.concatenate([[1.5*c[0] - 0.5*c[1]], 0.5*(c[1:] + c[:-1]), [1.5*c[-1] - 0.5*c[-2]]])


def signed_map(ax, x, y, F, vmax=None, label_zero=False, zero_label=r'$F = 0$',
               zero_at=None):
    """A signed quantity ``F[y, x]`` on seismic with white at zero, plus its zero contour.

    ``vmax`` shares a scale between panels; ``label_zero`` writes the contour
    label once (give ``zero_at=(x, y)`` to place it).  Returns the mesh, for
    the colour bar.
    """
    vmax = np.abs(F).max() if vmax is None else vmax
    mesh = ax.pcolormesh(edges(x), edges(y), F, cmap='seismic', vmin=-vmax, vmax=vmax,
                         shading='flat', rasterized=True)
    zero = ax.contour(x, y, F, levels=[0], colors='0.4', linestyles=':', linewidths=0.5)
    if label_zero:
        ax.clabel(zero, fmt={0: zero_label}, inline=True,
                  manual=[zero_at] if zero_at is not None else None)
    return mesh


def inset_colorbar(ax, mesh, title, box=(0.55, 0.12, 0.4, 0.05), fmt='%.2f'):
    """One horizontal colour bar inside the panel with the most empty space."""
    vmax = mesh.norm.vmax
    inset = ax.inset_axes(list(box))
    bar = ax.figure.colorbar(mesh, cax=inset, orientation='horizontal',
                             ticks=[-vmax, 0, vmax], format=fmt)
    bar.outline.set_linewidth(0.5)
    inset.set_title(title, pad=2)
    return bar


def save_raw(fig, results_dir, name, dpi=250):
    """``<results>/<name>_raw.png`` and ``.pdf``.  Never the bare name: that
    is the hand-finished figure."""
    results = Path(results_dir)
    results.mkdir(exist_ok=True)
    fig.savefig(results/f'{name}_raw.png', dpi=dpi)
    fig.savefig(results/f'{name}_raw.pdf', dpi=dpi)
    print(f'wrote {results/(name + "_raw.png")} and .pdf')
