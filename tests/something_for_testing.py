import numpy as np
import matplotlib.pyplot as plt

import utility
from utility import utils as u
from utility import cli_helper as helper
from utility.color import Painter
from utility.timer import Timer
from utility.templates.test_functions import TestFunction, TestFunctionType, Image, TestFunction2D
from utility.templates.base_transformations import Transformation
from transformations.walsh_transformation_1d.walsh_transformation_1d import WalshTransformation
from transformations.walsh_transformation_2d.walsh_transformation_2d import WalshTransformation2D
from transformations.wavelet_transformation_1d.wavelet_transformation_1d import WaveletTransformation
from transformations.wavelet_transformation_2d.wavelet_transformation_2d import WaveletTransformation2D
from transformations.discrete_cosine_transformation_1d.discrete_cosine_transformation_1d import \
    DiscreteCosineTransformation
from transformations.discrete_cosine_transformation_2d.discrete_cosine_transformation_2d import \
    DiscreteCosineTransformation2D


def inspect_cell(i: int, j: int, transformation: Transformation, coeffs: np.ndarray, extra_samples: int = 9):
    n = 2 ** transformation.n
    h = 1.0 / n
    x0, x1 = i * h, (i + 1) * h
    y0, y1 = j * h, (j + 1) * h
    c = transformation.sample_transform(coeffs, samples=n)[i, j]
    F = transformation.function.evaluate_integral
    integral_f = (F(x1, y1) - F(x0, y1) - F(x1, y0) + F(x0, y0))
    area = (x1 - x0) * (y1 - y0)
    print(f"cell ({i},{j}) x∈[{x0:.5g},{x1:.5g}], y∈[{y0:.5g},{y1:.5g}]")
    print("c_from_transform:", c)
    print("cell avg integral_f/area:", integral_f / area)
    print("per-cell L1:", abs(integral_f - c * area))
    # sample local grid
    xs = np.linspace(x0, x1, extra_samples)
    ys = np.linspace(y0, y1, extra_samples)
    X, Y = np.meshgrid(xs, ys)
    fvals = transformation.function.evaluate(X, Y)
    # reconstruct approx on same local grid using cell-constant c
    approx = np.full_like(fvals, c)
    diff = fvals - approx
    print("local f sample (corner,mid,corner):", fvals[0, 0], fvals[extra_samples // 2, extra_samples // 2],
          fvals[-1, -1])
    print("local max abs error:", np.max(np.abs(diff)))
    # optionally return arrays for plotting
    # return X, Y, fvals, approx


def per_cell_l1_map(transformation, coeffs):
    n_cells = 2 ** transformation.n
    h = 1.0 / n_cells
    F = transformation.function.evaluate_integral
    vals = transformation.sample_transform(coeffs, samples=n_cells)
    pc = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            x0, x1 = i * h, (i + 1) * h
            y0, y1 = j * h, (j + 1) * h
            area = (x1 - x0) * (y1 - y0)
            integral_f = (F(x1, y1) - F(x0, y1) - F(x1, y0) + F(x0, y0))
            c = float(vals[i, j])
            pc[i, j] = abs(integral_f - c * area)
    return pc


f = utility.test_functions_2d.Rational()
fine_trans = WalshTransformation2D(6, f)
fine_coeffs = fine_trans.get_coefficients_integration_orthonormal()

coarse_trans = WalshTransformation2D(2, f)
coarse_coeffs = coarse_trans.get_coefficients_integration_orthonormal()

# usage:
pc_coarse = per_cell_l1_map(coarse_trans, coarse_coeffs)
pc_fine = per_cell_l1_map(fine_trans, fine_coeffs)

# Upsample coarse map to fine resolution for cell-wise comparison:
scale = 2 ** (fine_trans.n - coarse_trans.n)
pc_coarse_up = np.repeat(np.repeat(pc_coarse, scale, axis=0), scale, axis=1) / (scale * scale)
delta = pc_fine - pc_coarse_up

print("pc_coarse.sum():", pc_coarse.sum())
print("pc_coarse_up.sum():", pc_coarse_up.sum())
print("pc_fine.sum():", pc_fine.sum())
print("difference (fine - coarse):", pc_fine.sum() - pc_coarse.sum())  # total change
print("difference (fine - coarse_up):", pc_fine.sum() - pc_coarse_up.sum())  # should be same as above

print("delta.sum():", delta.sum())
print("max delta:", delta.max())
print("min delta:", delta.min())
print("mean delta:", delta.mean())
print("positive deltas:", (delta > 0).sum(), "/", delta.size)
print("negative deltas:", (delta < 0).sum(), "/", delta.size)

# Top offenders
flat_idx = np.argsort(delta.ravel())[::-1]
print("\nTop 20 positive deltas:")
for k in range(20):
    idx = np.unravel_index(flat_idx[k], delta.shape)
    print(">",k, int(idx[0]), int(idx[1]), "delta:", delta[idx], "pc_fine:", pc_fine[idx], "pc_coarse_up:",
          pc_coarse_up[idx])
    inspect_cell(int(idx[0]), int(idx[1]), fine_trans, fine_coeffs)

flat_idx = np.argsort(delta.ravel())
print("\nTop 20 negative deltas:")
for k in range(20):
    idx = np.unravel_index(flat_idx[k], delta.shape)
    print(">",k, int(idx[0]), int(idx[1]), "delta:", delta[idx], "pc_fine:", pc_fine[idx], "pc_coarse_up:",
          pc_coarse_up[idx])
    inspect_cell(int(idx[0]), int(idx[1]), fine_trans, fine_coeffs)

res = 256
xs = np.linspace(0,1,res)
ys = np.linspace(0,1,res)
X,Y = np.meshgrid(xs, ys)
fgrid = f.evaluate(X,Y)

coarse_vals = coarse_trans.sample_transform(coarse_coeffs, samples=res)
fine_vals = fine_trans.sample_transform(fine_coeffs, samples=res)

f.plot()
fine_trans.plot_transformation(fine_vals,fgrid)
fine_trans.plot_error_absolute(fine_vals, fgrid)


plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("error coarse (f - fhat)"); plt.imshow(fgrid - coarse_vals, origin='lower'); plt.colorbar()
plt.subplot(1,3,2); plt.title("error fine (f - fhat)");   plt.imshow(fgrid - fine_vals, origin='lower');   plt.colorbar()
plt.subplot(1,3,3); plt.title("difference (fine - coarse)"); plt.imshow((fgrid - fine_vals) - (fgrid - coarse_vals), origin='lower'); plt.colorbar()
plt.show()