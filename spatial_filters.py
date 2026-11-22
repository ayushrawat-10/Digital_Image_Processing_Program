"""
=============================================================================
Module: spatial_filters.py
Description: Master Spatial Domain Filter Module (Imports from Individual Filters)
Course: Digital Image Processing (DIP)
=============================================================================

This module exposes all 7 spatial filters, each implemented in its own separate file:
- filter_mean.py               -> apply_mean_filter
- filter_weighted_average.py   -> apply_weighted_average_filter
- filter_gaussian.py           -> apply_gaussian_filter, generate_gaussian_kernel_3x3
- filter_median.py             -> apply_median_filter
- filter_laplacian.py          -> apply_laplacian_filter
- filter_sobel.py              -> apply_sobel_filter
- filter_high_boost.py         -> apply_high_boost_filter

Utility helpers:
- spatial_utils.py             -> pad_image, convolve2d_scratch
=============================================================================
"""

from spatial_utils import pad_image, convolve2d_scratch
from filter_mean import apply_mean_filter
from filter_weighted_average import apply_weighted_average_filter
from filter_gaussian import apply_gaussian_filter, generate_gaussian_kernel_3x3
from filter_median import apply_median_filter
from filter_laplacian import apply_laplacian_filter
from filter_sobel import apply_sobel_filter
from filter_high_boost import apply_high_boost_filter

__all__ = [
    'pad_image',
    'convolve2d_scratch',
    'apply_mean_filter',
    'apply_weighted_average_filter',
    'generate_gaussian_kernel_3x3',
    'apply_gaussian_filter',
    'apply_median_filter',
    'apply_laplacian_filter',
    'apply_sobel_filter',
    'apply_high_boost_filter'
]
