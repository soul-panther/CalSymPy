"""
CalSymPy - Heat Distribution Visualization
Team 4 (Delta-T)

What this does:
  - Defines 3 temperature models on a flat square plate: T(x, y)
      Model A = one hotspot in the middle
      Model B = edges held cold (fixed cooling)
      Model C = edges sealed (insulated, no heat escapes)
  - Uses SymPy to get the EXACT partial derivatives (no PDE, no time, syllabus-safe)
  - Uses NumPy to evaluate everything fast on a grid
  - Uses Matplotlib to draw heat maps + arrows showing which way heat flows
  - Runs some basic checks so we know the numbers are actually correct

Run it with:  python calsympy_heat_models.py
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. SET UP THE PLATE
# ----------------------------------------------------------------------
PLATE_LENGTH = 4.0   # cm, x goes from 0 to 4
PLATE_WIDTH = 4.0    # cm, y goes from 0 to 4
RESOLUTION = 100      # how many points along each side

x, y = sp.symbols('x y')

# ----------------------------------------------------------------------
# 2. THE 3 MODELS (plain formulas, no time, no PDE -> syllabus safe)
# ----------------------------------------------------------------------
# Model A: one hotspot in the centre (like one hot chip on a cool board)
T_A = 25 + 60 * sp.exp(-((x - 2)**2 + (y - 2)**2) / 2)

# Model B: edges held cold (like all 4 sides touching a cooling fan)
# sin(0) = sin(pi) = 0, so T = 20 on every edge automatically
T_B = 20 + 50 * sp.sin(sp.pi * x / 4) * sp.sin(sp.pi * y / 4)

# Model C: edges sealed, no heat escapes (insulated)
# cos has zero slope at x=0, x=4, y=0, y=4 -> no heat crosses the edge
T_C = 40 + 30 * sp.cos(sp.pi * x / 4) * sp.cos(sp.pi * y / 4)

MODELS = {
    "Model A - Single Hotspot": T_A,
    "Model B - Cold Edges": T_B,
    "Model C - Sealed Edges": T_C,
}


# ----------------------------------------------------------------------
# 3. FOR EACH MODEL: get exact derivatives (SymPy), then turn them into
#    fast numpy functions, then evaluate on the whole grid at once.
# ----------------------------------------------------------------------
def analyse_model(name, T_formula):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # --- exact calculus, done symbolically (this is the "by hand" part,
    #     just done by the computer instead of on paper) ---
    Tx = sp.diff(T_formula, x)          # how fast temp changes left-right
    Ty = sp.diff(T_formula, y)          # how fast temp changes up-down
    Txx = sp.diff(T_formula, x, 2)      # 2nd derivative (curvature), still just
    Tyy = sp.diff(T_formula, y, 2)      # ordinary partial derivatives, no PDE

    print("T(x, y)   =", T_formula)
    print("Tx (dT/dx) =", Tx)
    print("Ty (dT/dy) =", Ty)

    # --- turn the symbolic formulas into fast numpy functions ---
    T_func = sp.lambdify((x, y), T_formula, "numpy")
    Tx_func = sp.lambdify((x, y), Tx, "numpy")
    Ty_func = sp.lambdify((x, y), Ty, "numpy")

    # --- build the grid of points across the plate ---
    xs = np.linspace(0, PLATE_LENGTH, RESOLUTION)
    ys = np.linspace(0, PLATE_WIDTH, RESOLUTION)
    X, Y = np.meshgrid(xs, ys)

    # --- evaluate temperature + derivatives on the whole grid at once ---
    Z = T_func(X, Y) * np.ones_like(X)        # temperature everywhere
    dTx = Tx_func(X, Y) * np.ones_like(X)
    dTy = Ty_func(X, Y) * np.ones_like(X)

    # heat flows from HOT to COLD, which is the OPPOSITE of the gradient
    # (this is the mistake we made last time - fixed here)
    flow_x = -dTx
    flow_y = -dTy

    # --- quick sanity checks, so we know the numbers are right ---
    hottest_idx = np.unravel_index(np.argmax(Z), Z.shape)
    coldest_idx = np.unravel_index(np.argmin(Z), Z.shape)
    print(f"Hottest point: {Z[hottest_idx]:.2f} C at "
          f"x={X[hottest_idx]:.2f}, y={Y[hottest_idx]:.2f}")
    print(f"Coldest point: {Z[coldest_idx]:.2f} C at "
          f"x={X[coldest_idx]:.2f}, y={Y[coldest_idx]:.2f}")
    print(f"Average temperature: {Z.mean():.2f} C")

    return X, Y, Z, flow_x, flow_y


def plot_model(name, X, Y, Z, flow_x, flow_y, ax_heat, ax_contour):
    # Use the same temperature range for both views of this model.
    vmin = float(np.min(Z))
    vmax = float(np.max(Z))
    levels = np.linspace(vmin, vmax, 16)

    # heat map
    im = ax_heat.imshow(Z, extent=[0, PLATE_LENGTH, 0, PLATE_WIDTH],
                         origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    ax_heat.set_title("Heatmap", fontsize=11, pad=7)
    ax_heat.set_xlabel("x (cm)")
    ax_heat.set_ylabel("y (cm)")
    ax_heat.set_aspect("equal", adjustable="box")

    # contour map + arrows showing which way heat flows
    cs = ax_contour.contourf(X, Y, Z, levels=levels, cmap="inferno",
                             vmin=vmin, vmax=vmax)
    step = 10  # only draw every 10th arrow so it's not cluttered
    ax_contour.quiver(X[::step, ::step], Y[::step, ::step],
                       flow_x[::step, ::step], flow_y[::step, ::step],
                       color="white")
    ax_contour.set_title("Contours + heat flow", fontsize=11, pad=7)
    ax_contour.set_xlabel("x (cm)")
    ax_contour.set_ylabel("y (cm)")
    ax_contour.set_aspect("equal", adjustable="box")

    # One colorbar per model keeps the two views comparable without
    # inserting a separate colorbar gutter between every subplot.
    ax_heat.text(0.02, 0.98, name, transform=ax_heat.transAxes,
                 ha="left", va="top", fontsize=9, color="white",
                 bbox={"facecolor": "black", "alpha": 0.55, "pad": 3,
                       "edgecolor": "none"})
    ax_heat.figure.colorbar(cs, ax=[ax_heat, ax_contour], label="Temp (C)",
                            fraction=0.035, pad=0.025, aspect=30)


# ----------------------------------------------------------------------
# 4. RUN IT FOR ALL 3 MODELS AND PLOT THEM
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # One model per row makes the heatmap/contour comparison immediate.
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.2})
    fig.suptitle("Heat Distribution Models", fontsize=16)

    results = {}
    for i, (name, formula) in enumerate(MODELS.items()):
        X, Y, Z, flow_x, flow_y = analyse_model(name, formula)
        results[name] = Z
        plot_model(name, X, Y, Z, flow_x, flow_y, axes[i, 0], axes[i, 1])

    # quick comparison: does cooling the edges actually pull heat out?
    print("\n" + "=" * 60)
    print("COMPARISON: does cooling the edges matter?")
    print("=" * 60)
    avg_B = results["Model B - Cold Edges"].mean()
    avg_C = results["Model C - Sealed Edges"].mean()
    print(f"Model B (cold edges) average temp:   {avg_B:.2f} C")
    print(f"Model C (sealed edges) average temp: {avg_C:.2f} C")
    print("-> Cold edges pull the average DOWN. Sealed edges trap the heat.")

    plt.savefig("calsympy_results.png", dpi=180, bbox_inches="tight")
    print("\nSaved plot as calsympy_results.png")
    plt.show()
