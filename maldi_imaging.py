import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, MiniBatchKMeans


def load_matrix(path, sep='\t', **kwargs):
    """
    Loading matrix obtained after MALDIQuant library processing of .RAW files
    :param path: str - path to the matrix
    :param sep: str - field separator in file
    :return: df - pandas df with data
    """
    matrix = pd.read_csv(path, sep=sep, **kwargs)
    return matrix


def zeros(matrix):
    """
    Check whether matrix have some ions which intensity is 0 in all pixels or pixels which have 0 intensity in all ions
    :param matrix: df - pandas df with all data
    :return: raise AssertionError if conditions fail
    """
    # Zero elements in a matrix
    elem_is_zero = (matrix == 0)
    # Number of ions which intensities are 0 for pixels
    pixel_zero_num = elem_is_zero.sum(axis=1)

    # Which pixels have all intensities equal to 0
    is_blank_pixel = (pixel_zero_num == matrix.shape[1])

    # Remove blank pixels
    matrix = matrix.loc[~is_blank_pixel]

    # Number of pixels which have 0 intensity for ions
    mz_zero_num = elem_is_zero.sum()

    # Which ions have intensity equal to 0 in all pixels
    is_blank_mz = (mz_zero_num == matrix.shape[0])

    # Remove blank peaks
    matrix = matrix.loc[:, ~is_blank_mz]
    return matrix


def reindexing(matrix):
    """
    Create new index for matrix with the following structure - species, x, y
    Originally matrix should have index in a str form - 'x,y,species'
    Also convert column names to float
    :param matrix: df - pandas df with all data
    :return: df = pandas df with all data with new MultiIndex
    """
    # Make rearranged index with numeric coordinates
    matrix.index = pd.MultiIndex.from_tuples(list(map(
        lambda x: [int(x[i]) if x[i].isdigit() else str(x[i]) for i in [-1, 0, 1]],
        matrix.index.str.split(','))), names=['species', 'x', 'y'])
    # Convert column names from str to float
    matrix.columns = matrix.columns.astype(float)


def duplications(matrix):
    """
    Find duplicates in matrix by MultiIndex, i.e. same x, y and species values
    Find duplicates in matrix, i.e. same intensities in all pixels
    :param matrix: df - pandas df with all data with new MultiIndex
    :return: (df, df) - tuple with dfs - rows duplicated by index and fully duplicated rows
    """
    # Rows which are duplicates by index
    index_dups = matrix[matrix.index.duplicated()]
    print(f'There are {index_dups.shape[0]} duplicated by index rows')

    # Rows which are fully duplicates
    dups = matrix[matrix.duplicated()]
    print(f'There are {dups.shape[0]} fully duplicated rows')

    return index_dups, dups


def get_coords(matrix):
    """
    Extract coordinates from the matrix and convert them to 0-based coordinates
    :param matrix: df - pandas df with all data with new MultiIndex
    :return: (array, array) - pair of 1d np.arrays with 0-based coordinates of pixels in picture
    """
    # Extract 1-based x and y coordinates from index
    xs = matrix.index.get_level_values('x').values
    ys = matrix.index.get_level_values('y').values
    if min(xs) == 1 or min(ys) == 1:
        def base_one_to_zero(x):
            """Subtract 1 from argument"""
            return x - 1

        # Convert to 0-based indexing python convention
        xs = base_one_to_zero(xs)
        ys = base_one_to_zero(ys)

    return xs, ys


def draw(matrix, xs, ys, ion, save=False, path='img', format='png'):
    """
    Plot intensities of some ion in pixels on image, save image if required
    :param matrix: df - pandas df with all data with new MultiIndex
    :param xs: array - 1d np.arrays with 0-based x coordinates of pixels in picture
    :param ys: array - 1d np.arrays with 0-based y coordinates of pixels in picture
    :param ion: str/float - name of column from matrix corresponding to this ion
    :param save: bool - whether to save plotted graph, False by default
    :param path: str - path to directory to save images, 'img' by default
    :param format: str - format of image, 'png' by default
    :return:
    """
    # Create image, it have max(y) x max(x) size assuming that the pixels with greatest x and y coordinates are on
    # the edges of image
    image = np.zeros((ys.max() + 1, xs.max() + 1))
    # Fill pixel values with ion intensity
    # Remember that xs are column coordinates, and ys - row coordinates
    image[ys, xs] = matrix[ion]

    # Show image and add colorbar, this fraction is consistent with image height
    plt.imshow(image)
    # plt.colorbar(fraction=0.0257)

    # Make directories according to path and save picture in the leaf with name like specified ion.format
    if save:
        os.makedirs(path, exist_ok=True)
        plt.savefig(fname=f'{path}/{ion}.{format}', format=format)


def draw_batch(matrix, xs, ys, ions, path='img', format='png'):
    """
    Save image for each ion in ions
    :param matrix: df - pandas df with all data with new MultiIndex
    :param xs: array - 1d np.arrays with 0-based x coordinates of pixels in picture
    :param ys: array - 1d np.arrays with 0-based y coordinates of pixels in picture
    :param ions: iterable - collection with ions corresponding to matrix column names
    :param path: str - path to directory to save images
    :param format: str - format of image, 'png' by default
    :return:
    """
    # Create path
    os.makedirs(path, exist_ok=True)
    # Count number of rows and columns in image
    rows = ys.max() + 1
    cols = xs.max() + 1

    def _draw(matrix, rows, cols, ion, path):
        """Create and save 1 image, similar to previous function, perhaps refaсtor them"""
        image = np.zeros((rows, cols))
        image[ys, xs] = matrix[ion]
        plt.imshow(image)
        plt.colorbar(fraction=0.0257)
        plt.savefig(fname=f'{path}/{ion}.{format}', format=format)

    # Create and save picture for each ion
    for ion in ions:
        _draw(matrix, rows, cols, ion, path)


def compute_layout(plot_number, width_height=(6, 4)):
    """
    Compute how many rows and cols should be in plt.subplots layout, and figsize. It gives quasisquare shape
    :param plot_number: int - number of plots to place in 1
    :param width_height: (float, float) - width and height of 1 image, default is 6 and 4
    :return: (int, int, (float, float)) - number of rows, number of cols and figure size
    """
    # Get rough number of rows and columns (according to square shape)
    ncols = nrows = round(plot_number ** 0.5)

    # In case that rounded values give result less than number of plots increase number of rows
    if ncols * nrows < plot_number:
        nrows += 1

    # Make figsize proportional to the number of images in row and column
    figsize = (width_height[0] * ncols, width_height[1] * nrows)
    return nrows, ncols, figsize


def draw_panel(matrix, xs, ys, ions, width_height=(6, 4), save=False, path='img', format='png'):
    """
    Draw array of images with ion intensities from ions on picture as a panel
    :param matrix: df - pandas df with all data with new MultiIndex
    :param xs: array - 1d np.arrays with 0-based x coordinates of pixels in picture
    :param ys: array - 1d np.arrays with 0-based y coordinates of pixels in picture
    :param ions: iterable - collection with ions corresponding to matrix column names
    :param width_height: (float, float) - pair of width and height of each image on plot
    :param save: bool - whether to save plotted graph, False by default
    :param path: str - path to directory to save images
    :param format: str - format of image, 'png' by default
    :return:
    """
    # Get number of plots
    plotsize = ions.size

    # In most cases there are more than 1 ion
    if plotsize != 1:
        # Number of rows and columns in image
        rows = ys.max() + 1
        cols = xs.max() + 1

        # Get number of rows and columns in superplot, and general figure size
        nrows, ncols, figsize = compute_layout(plotsize, width_height)

        # Prepare subplots
        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        ax = ax.ravel()

        # Plot each image at superplot
        for i, ion in enumerate(ions):
            image = np.zeros((rows, cols))
            image[ys, xs] = matrix[ion]
            ax[i].imshow(image)
            ax[i].title.set_text(f'{ion}')

        # Create path and save image with the specified name panel_firstion-lastion.format if it's needed
        if save:
            os.makedirs(path, exist_ok=True)
            plt.savefig(f'{path}/panel_{ions[0]}-{ions[-1]}.{format}', format=format)
    # Draw 1 picture
    else:
        draw(matrix, xs, ys, ions[0], save, path, format)
    plt.close()


def compute_width_height(xs, ys):
    """
    Compute values of picture width and height
    :param xs: array - 1d np.arrays with 0-based x coordinates of pixels in picture
    :param ys: array - 1d np.arrays with 0-based y coordinates of pixels in picture
    :return: (float, float) - width and height of picture
    """
    # 22.5 is just an empiric observation
    return xs.max() / 22.5, ys.max() / 22.5


def draw_species_panels(matrix_path, path, species='h', span=(500, 1501), format='png'):
    """
    Draw array of images with ion intensities from ions on picture as a panel for all ion groups for specified species
    :param matrix_path: str - path to the matrix
    :param path: str - path to directory to save images
    :param species: str - species to select from the matrix
    :param span: tuple - from which to which
    :param format: str - format of image, 'png' by default
    :return:
    """
    # Load matrix
    matrix = load_matrix(matrix_path, sep='\t')

    # Zero pixels
    zeros(matrix)
    # Multiindex adding
    reindexing(matrix)
    # # of duplicates
    duplications(matrix)

    # Take only 1 species
    selected_data = matrix.query(f'species == "{species}"')

    # Pixel coordinates
    xs, ys = get_coords(selected_data)

    # Width and height of each panel
    width_height = compute_width_height(xs, ys)

    # Iterate over ion groups in df columns and draw panel of them
    for i, ion_group in enumerate(range(*span)):
        if i % 10 == 0:
            print(f'{ion_group} processing')

        # Take ions
        ions = selected_data.filter(regex=rf'^{ion_group}.*').columns
        # Don't draw if there is no ions in this group
        if ions.size:
            draw_panel(selected_data, xs, ys, ions, width_height=width_height, save=True, path=path, format=format)


def reindexed_draw_species_panels(matrix_path, path, species='h', span=(500, 1501), format='png'):
    """
    Draw array of images with ion intensities from ions on picture as a panel for all ion groups for specified species.
    Use it for already reindexed matrices
    :param matrix_path: str - path to the reindexed matrix
    :param path: str - path to directory to save images
    :param species: str - species to select from the matrix
    :param span: tuple - from which to which
    :param format: str - format of image, 'png' by default
    :return:
    """
    # Load matrix
    matrix = load_matrix(matrix_path, sep='\t', index_col=[0, 1, 2])

    # Take only 1 species
    selected_data = matrix.query(f'species == "{species}"')

    # Pixel coordinates
    xs, ys = get_coords(selected_data)

    # Width and height of each panel
    width_height = compute_width_height(xs, ys)

    # Iterate over ion groups in df columns and draw panel of them
    for i, ion_group in enumerate(range(*span)):
        if i % 10 == 0:
            print(f'{ion_group} processing')

        # Take ions
        ions = selected_data.filter(regex=rf'^{ion_group}.*').columns
        # Don't draw if there is no ions in this group
        if ions.size:
            draw_panel(selected_data, xs, ys, ions, width_height=width_height, save=True, path=path, format=format)


def mz_intensity_plot(mz_intensity, ion_range=None, save=False, path='img', format='png'):
    """
    Draw plot of mz vs intensity of all ions or part of them
    :param mz_intensity: series - series of mzs and intensities
    :param ion_range: (float, float) - pair of start value and stop for x axis
    :param save: bool - whether to save plotted graph, False by default
    :param path: str - path to directory to save images
    :param format: str - format of image, 'png' by default
    :return:
    """
    # Make figure large enough
    plt.figure(figsize=(12, 8))

    # Plot mz vs intensity of part of ions or of all ions
    if ion_range:
        sns.lineplot(data=mz_intensity.loc[:, ion_range[0]:ion_range[1]])
    else:
        sns.lineplot(data=mz_intensity)

    # Create path and save image with the specified name mz_intensity_firstion-lastion.format if it's needed
    if save:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/mz_intensity_{ion_range[0]}-{ion_range[-1]}.{format}', format=format)


def pixel_stats(matrix, threshold):
    """
    Compute pixel intensity stats, fraction of pixels on slice with intensity higher than threshold and similar fraction
    on the whole image
    :param matrix: df - pandas df with all data with new MultiIndex
    :param threshold: number - minimal intensity for pixel to be counted as valid
    :return: (series, float, float) - pixel intensity stats, fraction of intense enough pixels on slice and
    fraction of intense enough pixels on whole picture
    """
    # Get intensity of all mz for each pixel
    pixel_intensity = matrix.sum(axis=1)

    # Get number of pixels with overall intensity greater than threshold
    number_of_intense_pixels = (pixel_intensity >= threshold).sum()
    number_of_pixels = pixel_intensity.size
    # Pixel intensity stats, fraction of intense enough pixels on slice, same on whole picture
    return (pixel_intensity.describe(),
            number_of_intense_pixels / number_of_pixels,
            number_of_intense_pixels / (
                    matrix.index.get_level_values('x').max() * matrix.index.get_level_values('y').max()))


def difference(ion1, ion2):
    """
    Compute relative difference (ppm)
    :param ion1: float - mz of ion
    :param ion2: float - mz of ion
    :return: float - relative difference between ion1 and ion2
    """
    return abs(ion1 - ion2) / ion1


def absolute_recalibration(mzs, shift):
    """
    Recalibrate mz by specified value - mz + shift, e.g. 500 + 2, 1000 + 2
    :param mzs: number or series - mz which will be recalibrated
    :param shift: number - daltons which will be added to mzs
    :return: recalibrated mzs
    """
    return mzs + shift


def relative_recalibration(mzs, ppm):
    """
    Recalibrate mz by ppm - mz / (1 - ppm), e.g. 500 + 0.01, 1000 + 0.02
    :param mzs: number or series - mz which will be recalibrated
    :param ppm: number - ppm which will be added to mzs
    :return: recalibrated mzs
    """
    return mzs / (1 - ppm * 1e-6)


def merge_peaks(ion1, ion2, intensity1, intensity2):
    """
    Merge 2 peaks together, averaging their mz and summing intensity
    :param ion1: float - mz of ion
    :param ion2: float - mz of ion
    :param intensity1: float - intensity of ion
    :param intensity2: float - intensity of ion
    :return: (float, float) - pair of new mz and intensity
    """
    # Get middle mz
    mz = (ion1 + ion2) / 2
    # Sum intensities
    intensity = intensity1 + intensity2
    return mz, intensity


def merge_series(mzs, ppm):
    """
    Merge all peaks in mzs which have mz difference less than ppm, summing their intensities
    :param mzs: series - series with intensity for each mz
    :param ppm: float - accepted difference between 2 mzs to be treated as 1
    :return: series - series with merged mz and intensities
    """
    # Initialize list with new mzs and intensities
    merged = [mzs.index[0]]
    intensities = [mzs.values[0]]
    # Scale ppm to fraction
    ppm *= 1e-6

    # Looks like it can be rewritten with rolling window(2)
    # Iterate over rest of peaks, merge each of them with previous obtained if difference in mz less than ppm
    # Otherwise add peak intact to new peaks
    for mz, current_int in mzs.iloc[1:].iteritems():
        if difference(merged[-1], mz) <= ppm:
            merged[-1], intensities[-1] = merge_peaks(merged[-1], mz, intensities[-1], current_int)
        else:
            merged.append(mz)
            intensities.append(current_int)
    return pd.Series(data=intensities, index=merged)


def merge_df(matrix, ppm):
    """
    Merge all peaks in matrix which have mz difference less than ppm, summing their intensities
    :param matrix: df - df with all data
    :param ppm: float - accepted difference between 2 mzs to be treated as 1
    :return: df - df with merged mz and intensities
    """
    # Initialize list with new mzs and intensities
    merged = [matrix.columns[0]]
    intensities = [matrix.iloc[:, 0]]
    # Scale ppm to fraction
    ppm *= 1e-6

    # Iterate over rest of peaks, merge each of them with previous obtained if difference in mz less than ppm
    # Otherwise add peak intact to new peaks
    for mz in matrix.columns[1:]:
        current_int = matrix[mz]

        if difference(merged[-1], mz) <= ppm:
            merged[-1], intensities[-1] = merge_peaks(merged[-1], mz, intensities[-1], current_int)
        else:
            merged.append(mz)
            intensities.append(current_int)
    return pd.DataFrame(data=dict(zip(merged, intensities)))


def merge(matrix, ppm):
    """
    Merge all peaks which have mz difference less than ppm, summing their intensities
    :param matrix: df - df with all data
    :param ppm: float - accepted difference between 2 mzs to be treated as 1
    :return: df - df with merged mz and intensities
    """
    # Choose appropriate function depending on type of passed object
    if isinstance(matrix, pd.DataFrame):
        return merge_df(matrix, ppm)
    return merge_series(matrix, ppm)


def matrix_to_mz_intensities(matrix):
    """
    Convert matrix to a series with summary intensity over each mz
    :param matrix: df - df with all data
    :return: series - series with intensity for each mz
    """
    return matrix.sum()


def load_maldi_lc(lcms, matrix):
    """
    Load LC dataset and MALDI dataset
    # todo - obsolete due to already written reindexed maldi matrix, rewrite
    # todo can preprocess lc (mz as index) and maldi (columns to float type)
    :param lcms: str - path to LC dataset
    :param matrix: str - path to MALDI dataset
    :return: (df, df) - tuple with LC dataframe and reindexed MALDI dataframe
    """
    lcms = pd.read_csv(lcms, sep='\t', index_col=0)
    matrix = pd.read_csv(matrix, sep='\t')

    # Multiindex
    reindexing(matrix)
    return lcms, matrix


def naive_align_peaks(lcms, matrix, threshold=5):
    """
    Align peaks from MALDI matrix on LCMS
    :param lcms: df - df with LC data, where mz is an index
    :param matrix: df - df with all MALDI data, where columns are mz floats
    :param threshold: float - number of difference between MALDI and LC peaks mz in ppm, which can be still aligned
    :return: df, dict - MALDI matrix with aligned renamed peaks and dictionary with correspondence between old MALDI
    and LC peak
    """
    # Convert ppm to fraction
    threshold *= 1e-6

    # Get relative differences between each ion in LC and MALDI
    lc_diffs = {lc: {(m, difference(lc, m)) for m in matrix.columns} for lc in lcms.index}
    # Get only mz with minimal ppm difference - will contains only 1 variant for ion
    lc_diffs = {lc: min(diffs, key=lambda x: x[1]) for lc, diffs in lc_diffs.items()}
    # Filter out peaks with relative difference > than 5ppm
    lc_diffs = {lc: diff for lc, diff in lc_diffs.items() if diff[1] <= threshold}

    # Dictionary with correspondence between MALDI and LC aligned peaks
    renaming = {m: lc for lc, (m, _) in lc_diffs.items()}
    print(f'Number of aligned peaks is {len(renaming)}')

    # Make peak mz consistent in MALDI and LC
    aligned_matrix = matrix.rename(columns=renaming)
    aligned_matrix = aligned_matrix[list(sorted(renaming.values()))]
    return aligned_matrix, renaming


def get_ratio(intensities1, intensities2):
    """
    Get ratio of corresponding log transformed intensities from intensities1 and intensities2
    :param intensities1: iterable - collection with intensities in a form of series or scalar
    :param intensities2: iterable - collection with intensities in a form of series or scalar
    :return: list - list with ratios of corresponding intensities
    """
    ratios = []

    # Iterate over each intensity and subtract 2nd from the 1st
    for ints1, ints2 in zip(intensities1, intensities2):
        ratios.append(ints1 - ints2)
    return ratios


def get_correlations(intensities1, intensities2):
    """
    Compute Pearson and Spearman correlations of corresponding intensities from intensities1 and intensities2
    :param intensities1: series - series with intensities
    :param intensities2: series - series with intensities
    :return: list - list with Pearson and Spearman correlations between pairs of intensity series
    """
    corrs = []

    # Compute correlation for corresponding intensities
    for ints1, ints2 in zip(intensities1, intensities2):
        corrs.extend((ints1.corr(ints2, method='pearson'),
                      ints1.corr(ints2, method='spearman')))
    return corrs


def subsetting_mean(lcms, matrix, renaming):
    # todo generalize
    """
    Divide MALDI and LC by species, return mean of each group
    :param lcms: df - df with lc data, where mz is an index
    :param matrix: df - df with all maldi data
    :return: (series, series, series, series, series, series) - tuple of series with mean for each group
    """

    # Get species subsets of LC
    human_lc = lcms.filter(regex=r'_H[ABCD]')
    chimp_lc = lcms.filter(regex=r'_CH[ABCD]')
    macaque_lc = lcms.filter(regex=r'_M[ABCD]')

    # Create an iterable with LC subsets
    lcs = [human_lc, chimp_lc, macaque_lc]
    lcs_avg = []

    for ls in lcs:
        # Compute mean for each peak
        ls_mean = ls.mean(axis=1)

        # Get only aligned peaks
        aligned = ls_mean[ls_mean.index.isin(renaming.values())]
        lcs_avg.append(aligned)

    # Get average intensities of each species
    human_lca, chimp_lca, macaque_lca = lcs_avg

    # Get species subsets
    human_maldi = matrix.query('species == "h"')
    chimp_maldi = matrix.query('species == "c"')
    macaque_maldi = matrix.query('species == "m"')

    # Take mean for each peak in maldi
    human_maldi = human_maldi.mean()
    chimp_maldi = chimp_maldi.mean()
    macaque_maldi = macaque_maldi.mean()
    return human_lca, chimp_lca, macaque_lca, human_maldi, chimp_maldi, macaque_maldi


def align(lcms, maldi):
    """
    Align peaks from MALDI on LCMS, get species and compute correlation of human/chimp and human/macaque ratios between
    LC and MALDI
    :param lcms: df - df with LC data
    :param maldi: df - df with all MALDI data
    :return: tuple - (list, int) with human/chimp and human/macaque correlations between LC and MALDI and
    number of aligned peaks
    """
    # Align peaks
    aligned_matrix, renaming = naive_align_peaks(lcms, maldi)

    # Find minimum
    pseudo = aligned_matrix[aligned_matrix > 0].min().min()
    # Add minimum to get rid of zeros
    aligned_matrix += pseudo

    # Log transform MALDI data
    aligned_matrix = logarithmize_maldi(aligned_matrix)

    # Separate species and compute mean for them
    human_lca, chimp_lca, macaque_lca, human_maldi, chimp_maldi, macaque_maldi = \
        subsetting_mean(lcms, aligned_matrix, renaming)

    # Get ratios of species intensities
    human_chimp_maldi, human_macaque_maldi, human_chimp_lc, human_macaque_lc = get_ratio(
        [human_maldi, human_maldi, human_lca, human_lca],
        [chimp_maldi, macaque_maldi, chimp_lca, macaque_lca])

    # Correlations between data subsets and number of aligned peaks
    return get_correlations([human_chimp_maldi, human_macaque_maldi], [human_chimp_lc, human_macaque_lc]), len(renaming)


def logarithmize_maldi(matrix):
    """
    Log2 transformation of matrix
    :param matrix: df - dataframe with intensity data
    :return: df - log transformed df
    """
    return np.log2(matrix)


def recalibrate_align(lcms, maldi, span, method='absolute'):
    """
    Compute Pearson and Spearman correlations between MALDI and LC with different MALDI recalibrations
    :param lcms: df - df with LC data
    :param maldi: df - df with all MALDI data
    :param span: tuple - minimal and maximal shifts and step in MALDI mz
    :param method: str - method to use for mz recalibration - one of {'absolute', 'relative'}, absolute is default
    :return: list - list with Pearson and Spearman correlations and number of aligned peaks for all shifts
    """
    # Prepare container
    correlations = []
    # Save original mz values
    original_mz = maldi.columns

    # For each mz shift recalibrate MALDI and compute correlations
    for i in np.arange(*span):
        # Recalibrate
        maldi.columns = recalibration[method](original_mz, i)
        # Align peaks
        corr, npeaks = align(lcms, maldi)

        # Also count number of peaks
        corr.append(npeaks)
        correlations.append(corr)

    # Restore original columns in maldi dataset
    maldi.columns = original_mz
    return correlations


# From method name to function for mz recalibration
recalibration = {'absolute': absolute_recalibration,
                 'relative': relative_recalibration}

# From RGB hex to decimal values
to_rgb = {'#DC143C': [220, 20, 60],
          '#FFD700': [255, 215, 0],
          '#BDB76B': [189, 183, 107],
          '#006400': [0, 100, 0],
          '#2E8B57': [46, 139, 87],
          '#000080': [0, 0, 128],
          '#483D8B': [72, 61, 139],
          '#800080': [128, 0, 128],
          '#556B2F': [85, 107, 47],
          '#1E90FF': [30, 144, 255],
          '#4B0082': [75, 0, 130],
          '#008080': [0, 128, 128],
          '#00FF00': [0, 255, 0]}

# From RGB hex to normalized values
to_rgb = {k: [v / 255 for v in vs] for k, vs in to_rgb.items()}

# Set of RGB colors to different clusters
colors = {2: ['#DC143C', '#FFD700'],
          3: ['#DC143C', '#FFD700', '#BDB76B'],
          4: ['#DC143C', '#FFD700', '#BDB76B', '#006400'],
          5: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57'],
          6: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#008080'],
          7: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#008080', '#483D8B'],
          8: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#000080', '#483D8B', '#800080'],
          9: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#000080', '#483D8B', '#800080', '#556B2F'],
          10: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#000080', '#483D8B', '#800080', '#556B2F',
               '#1E90FF'],
          11: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#000080', '#483D8B', '#800080', '#556B2F',
               '#1E90FF', '#4B0082'],
          12: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#000080', '#483D8B', '#800080', '#556B2F',
               '#1E90FF', '#4B0082', '#008080'],
          13: ['#DC143C', '#FFD700', '#BDB76B', '#006400', '#2E8B57', '#000080', '#483D8B', '#800080', '#556B2F',
               '#1E90FF', '#4B0082', '#008080', '#00FF00']}
# crimson, gold, dark khaki, dark green, seagreen, navy, darkslateblue, purple, darkolivegreen, dodgerblue, indigo, teal, lime
# Set of normalized rgb values
colors_rgb = {k: [to_rgb[v] for v in vs] for k, vs in colors.items()}


def kmeans_clustering(data, n=12):
    """
    Cluster data with k-means algorithm on [2, n) clusters, return dictionary with cluster labels
    :param data: df - dataframe with selected data (i.e. from 1 species)
    :param n: int - maximum number of clusters to divide
    :return: dict - dictionary in form of {cluster_number: [cluster_labels]}
    """
    # Dictionary with cluster labels for each cluster
    clusters = {}

    # Clustering for each number of clusters, get label of each pixel
    for i in range(2, n):
        print(f'\rprocessing separation on {i} clusters', end='')
        # Cluster and assign labels to dict
        clusters[i] = KMeans(n_clusters=i).fit_predict(data)
    print()
    return clusters


def _auxiliary(data, n=12):
    """
    Compute parameters necessary for plotting
    :param data: df - data, used for clustering, which going to be plotted
    :param n: int - maximum number of clusters to divide
    :return: (xs, ys, nrows, ncols, figsize, rows, cols) - tuple with parameters necessary for panel draw
    """
    # Auxiliary preparation
    xs, ys = get_coords(data)
    width_height = compute_width_height(xs, ys)
    nrows, ncols, figsize = compute_layout(n - 2, width_height)
    rows = ys.max() + 1
    cols = xs.max() + 1
    return xs, ys, nrows, ncols, figsize, rows, cols


def _draw_clusters(clusters, xs, ys, nrows, ncols, figsize, rows, cols):
    """
    Draw clusterizations on 1 plot
    :param clusters: dict - dictionary with cluster labels in form of {cluster_number: [cluster_labels]}
    :param xs: array - 1d np.arrays with 0-based x coordinates of pixels in picture
    :param ys: array - 1d np.arrays with 0-based y coordinates of pixels in picture
    :param nrows: int - number of rows
    :param ncols: int - number of cols
    :param figsize: (float, float) - tuple with figure size
    :param rows: int - number of rows in matrix
    :param cols: int - number of cols in matrix
    :return:
    """
    # Prepare subplots
    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    ax = ax.ravel()

    # Draw each separation on i clusters with its color scheme
    for i, cluster in clusters.items():
        image = np.zeros((rows, cols, 3))
        image[ys, xs] = colorize(image[ys, xs], cluster, colors_rgb[i])

        # For concordance between subplot and clusteriation subplot
        # 2 is a minimum of cluster number
        ax[i - 2].imshow(image)
        ax[i - 2].title.set_text(f'{i} clusters k-means')


def colorize(image, cluster, colors):
    """
    Paint image
    :param image: array - np 1d array subset of image matrix (3d) with pixels used in clusterization
    (their number should be the same as cluster size)
    :param cluster: array - np 1d array with cluster labels
    :param colors: list - list with normalized rgb color values, like [[0, 0.125, 0.5], ...]
    :return: colored subset of image matrix with pixels used in clusterization (values corresponds to a color now)
    """
    # For all clusters assign its pixel value to distinct rgb color
    for c in np.unique(cluster):
        image[cluster == c] = colors[c]
    return image


def draw_clusters(data, path, name, n=12, format='png'):
    """
    Cluster data with k-means on different number of clusters and plot these variants on 1 figure
    :param data: df - data, used for clustering, which going to be plotted
    :param path: str - path to folder where figure will be stored
    :param name: str - name of figure file
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    # Cluster data with cluster number [2, n)
    clusters = kmeans_clustering(data, n)

    # Take 1st letter
    species = name.split('_')[1][0]

    os.makedirs(path, exist_ok=True)
    # Write clusterization to file
    with open(f'{path}/area_clusters_{species_to_name[species]}.json', 'w') as dest:
        json.dump({cluster: labels.tolist() for cluster, labels in clusters.items()}, dest)

    # Preparations to plotting
    xs, ys, nrows, ncols, figsize, rows, cols = _auxiliary(data, n)
    # Plotting
    _draw_clusters(clusters, xs, ys, nrows, ncols, figsize, rows, cols)

    # Save figure and close everything
    plt.savefig(f'{path}/{name}', format=format)
    plt.close()


# Mapping from letter to species
species_to_name = {'h': 'human',
                   'c': 'chimp',
                   'm': 'macaque'}


def draw_area_clusters(files, n=12, format='png', **kwargs):
    """
    Draw k-mean clusterization with number of clusters from 2 to n on 1 plot
    :param files: iterable - collection with full paths to a matrix files
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    for file in files:
        # Load data
        matrix = load_matrix(file, **kwargs)
        print(f'Loaded {file}')

        # Get name of file without extension
        file = file.split('/')[-1].split('.')[0]

        # Zero pixels
        zeros(matrix)
        # Multiindex
        reindexing(matrix)

        for species in ['h', 'c', 'm']:
            # Get data for 1 species
            subset = matrix.query(f'species == "{species}"')
            print(f'Working with {species} subset')

            # Clustering
            draw_clusters(subset, f'images/{file}/clusters/area/',
                          name=f'kmeans_{species_to_name[species]}_clusters.{format}',
                          n=n, format=format)


def draw_clean_area_clusters(files, n=12, format='png', **kwargs):
    """
    Draw k-mean clusterization with number of clusters from 2 to n on 1 plot
    :param files: iterable - collection with full paths to a matrix files
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    for file in files:
        # Load data
        matrix = load_matrix(file, **kwargs)
        print(f'Loaded {file}')

        # Get name of file without extension
        file = file.split('/')[-1].split('.')[0]

        # Zero pixels
        zeros(matrix)

        for species in ['h', 'c', 'm']:
            # Get data for 1 species
            subset = matrix.query(f'species == "{species}"')
            print(f'Working with {species} subset')

            # Clustering
            draw_clusters(subset, f'cleaned_images/{file}/clusters/area/',
                          name=f'kmeans_{species_to_name[species]}_clusters.{format}',
                          n=n, format=format)


def draw_peak_clusters(files, n=12, format='png'):
    """
    Draw peak k-mean clusterization with number of clusters from 2 to n. There is a picture with all
    :param files: iterable - collection with full paths to a matrix files
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    for file in files:
        # Load data
        matrix = load_matrix(file)
        file = file.split('/')[-1].split('.')[0]
        print(f'Loaded {file}')
        # Zero pixels
        zeros(matrix)
        # Multiindex
        reindexing(matrix)

        # Create directory
        os.makedirs(f'images/{file}/clusters/peaks', exist_ok=True)

        # For each species clusterize peaks and draw a picture with clusterization
        for species in ['h', 'c', 'm']:
            # Take 1 species
            subset = matrix.query(f'species == "{species}"')
            print(f'Working with {species} subset')

            # Clustering
            clusters = kmeans_clustering(subset.T, n)
            # Write clusterization to file
            with open(f'images/{file}/clusters/peaks/peak_clusters_{species_to_name[species]}.json', 'w') as dest:
                json.dump({cluster: labels.tolist() for cluster, labels in clusters.items()}, dest)

            # Get necessary parameters for plotting
            xs, ys, rows, cols, width_height = light_plot_preparation(subset)

            # For each cluster create superplot
            for i, cluster in clusters.items():
                nrows, ncols, figsize = compute_layout(i, width_height)
                f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
                ax = ax.ravel()

                # Draw each subplot
                for j in range(i):
                    image = np.zeros((rows, cols))
                    image[ys, xs] = subset.loc[:, cluster == j].sum(axis=1)
                    # 2 - minima of cluster number
                    ax[j].imshow(image)
                    ax[j].title.set_text(f'{i} clusters k-means\ncluster #{j}')
                plt.savefig(f'images/{file}/clusters/peaks/kmeans_{species_to_name[species]}_clusters_{i}.{format}',
                            format=format)
    plt.close('all')


def reindexed_draw_peak_clusters(files, n=12, format='png'):
    """
    Draw peak k-mean clusterization with number of clusters from 2 to n, variant for reindexed matrices.
    There is a picture with all
    :param files: iterable - collection with full paths to a matrix files
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    for file in files:
        # Load data
        matrix = load_matrix(file)
        file = file.split('/')[-1].split('.')[0]
        print(f'Loaded {file}')

        # Create directory
        os.makedirs(f'images/{file}/clusters/peaks', exist_ok=True)

        # For each species clusterize peaks and draw a picture with clusterization
        for species in ['h', 'c', 'm']:
            # Take 1 species
            subset = matrix.query(f'species == "{species}"')
            print(f'Working with {species} subset')

            # Clustering
            clusters = kmeans_clustering(subset.T, n)
            # Write clusterization to file
            with open(f'images/{file}/clusters/peaks/peak_clusters_{species_to_name[species]}.json', 'w') as dest:
                json.dump({cluster: labels.tolist() for cluster, labels in clusters.items()}, dest)

            # Get necessary parameters for plotting
            xs, ys, rows, cols, width_height = light_plot_preparation(subset)

            # For each cluster create superplot
            for i, cluster in clusters.items():
                nrows, ncols, figsize = compute_layout(i, width_height)
                f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
                ax = ax.ravel()

                # Draw each subplot
                for j in range(i):
                    image = np.zeros((rows, cols))
                    image[ys, xs] = subset.loc[:, cluster == j].sum(axis=1)
                    # 2 - minima of cluster number
                    ax[j].imshow(image)
                    ax[j].title.set_text(f'{i} clusters k-means\ncluster #{j}')
                plt.savefig(f'images/{file}/clusters/peaks/kmeans_{species_to_name[species]}_clusters_{i}.{format}',
                            format=format)
    plt.close('all')


def light_plot_preparation(data):
    """
    Helper to get necessary for plotting information
    :param data: df - pandas df with all data with new MultiIndex
    :return: tuple - xs and ys coordinates, number of rows and columns in a matrix and width and height of image
    """
    # Plot preparation
    xs, ys = get_coords(data)
    rows = ys.max() + 1
    cols = xs.max() + 1
    width_height = compute_width_height(xs, ys)
    return xs, ys, rows, cols, width_height


def load_clusters(path):
    """
    Load dict with clusterization in json form from file
    :param path: str - path to jsoned clustering
    :return: {cluster: labels} - dict with clustering information
    """
    # Load json
    with open(path) as dest:
        clusters = json.load(dest)

    # Convert cluster number to int and lists to np.array
    clusters = {int(cluster): np.array(labels) for cluster, labels in clusters.items()}
    return clusters


def dump_dirt(dirt_peaks, more_intense_in_probe, path='../matrices/dirt/run4_p2/macaque.json'):
    """
    Write dirty mz to a json
    :param dirt_peaks: Float64Index - mz of peaks from dirt clusters
    :param more_intense_in_probe: series - boolean series with mz index of original df
    :param path: str - where to save file
    :return:
    """
    # Get mz of dirt peaks, who too intense in a matrix
    intensive_dirt_peaks = more_intense_in_probe.index[~more_intense_in_probe]

    # Unite peaks from dirt clusters and those who too intense in a matrix
    all_dirt = intensive_dirt_peaks.union(dirt_peaks)

    # Write dirt to a file
    with open(path, 'w') as dest:
        json.dump(all_dirt.tolist(), dest)


def mask_dirt_area(matrix, dirt_clusters, clustering):
    """
    Compose mask to get clean area
    :param matrix: df - dataframe with 1 species
    :param dirt_clusters: iterable - collection with # of dirt clusters
    :param clustering: array - 1d int array with cluster labels
    :return: array - 1d boolean array where True corresponds to clean pixels
    """
    # All pixels are initialized as clean
    p = np.ones(matrix.shape[0], dtype=bool)
    # Obtain mask for clean area, that is mask out dirt
    for c in dirt_clusters:
        p ^= clustering == c
    return p


def mask_dirt_peaks(matrix, dirt_clusters, clustering):
    """
    Compose mask to get clean peaks
    :param matrix: df - dataframe with 1 species
    :param dirt_clusters: iterable - collection with # of dirt clusters
    :param clustering: array - 1d int array with cluster labels
    :return: array - 1d boolean array where True corresponds to clean peaks
    """
    # All peaks are initialized as clean
    p = np.ones(matrix.shape[1], dtype=bool)
    # Obtain mask for clean peaks, that is mask out dirt
    for c in dirt_clusters:
        p ^= clustering == c
    return p


def dummy_peak_draw(matrix, n=1, clustering=None):
    """
    Draw picture of matrix or panel with peak clusters with summed peak intensities
    :param matrix: df - dataframe with 1 species
    :param n: int - number of clusters
    :param clustering: dict - dictionary with peak clustering scheme
    :return:
    """
    # Get necessary parameters for plotting
    xs, ys, rows, cols, width_height = light_plot_preparation(matrix)

    # Draw each subplot
    if clustering:
        # Preparation
        nrows, ncols, figsize = compute_layout(n, width_height)
        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        ax = ax.ravel()

        for j in range(n):
            image = np.zeros((rows, cols))
            image[ys, xs] = matrix.loc[:, clustering[n] == j].sum(axis=1)
            ax[j].imshow(image)
            ax[j].set_title(f'cluster #{j}', {'fontsize': 35})

    # Draw summary intensity
    else:
        image = np.zeros((rows, cols))
        image[ys, xs] = matrix.sum(axis=1)
        plt.imshow(image)


def clean_area(matrix, dirt_area_clusters, clustering):
    """
    Clean matrix from dirty pixels whose clusters are in dirt_area_clusters and from dirty peaks which are
    too intense in matrix
    :param matrix: df - df with 1 species
    :param dirt_area_clusters: iterable - collection with # of dirt clusters
    :param clustering: array - 1d int array with cluster labels
    :return: df, series - df with cleaned area and boolean series with marks about peaks intensity in 2 zones
    """
    # Get info whether pixels are clean
    cleaned_area = mask_dirt_area(matrix, dirt_area_clusters, clustering)

    # Get rid of dirt area
    area_cleaned = matrix.loc[cleaned_area]
    area_dirt = matrix.loc[~cleaned_area]

    # Find peaks which are more intense in sample than in matrix
    more_intense_in_probe = 10 * area_dirt.mean() < area_cleaned.mean()

    # Filter out matrix peaks
    area_cleaned = area_cleaned.loc[:, more_intense_in_probe]
    return area_cleaned, more_intense_in_probe


def clean_peaks(matrix, dirt_peak_clusters, clustering):
    """
    Clean matrix from dirty peaks whose clusters are in dirt_peak_clusters
    :param matrix: df - df with 1 species
    :param dirt_area_clusters: iterable - collection with # of dirt clusters
    :param clustering: array - 1d int array with cluster labels
    :return: index, index - mz of clean peaks and mz of dirt peaks
    """
    # Get info whether peaks are clean
    is_clean_peaks = mask_dirt_peaks(matrix, dirt_peak_clusters, clustering)

    # Get clean peaks
    peaks_cleaned = matrix.columns[is_clean_peaks]

    # Get dirt peaks
    dirt_peaks = matrix.columns[~matrix.columns.isin(peaks_cleaned)]

    return peaks_cleaned, dirt_peaks


def dummy_area_draw(matrix, n=1, clustering=None):
    """
    Draw picture of matrix or panel with area clusters
    :param matrix: df - dataframe with 1 species
    :param n: int - number of clusters
    :param clustering: dict - dictionary with area clustering scheme
    :return:
    """
    # Get necessary parameters for plotting
    xs, ys, rows, cols, width_height = light_plot_preparation(matrix)

    # Draw each subplot
    if clustering:
        # Preparation
        nrows, ncols, figsize = compute_layout(n, width_height)
        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        ax = ax.ravel()

        for j in range(n):
            # xs and ys are different each time
            xs, ys, rows, cols, _ = light_plot_preparation(matrix.loc[clustering[n] == j])
            # Create image
            image = np.zeros((rows, cols))
            image[ys, xs] = 1
            ax[j].imshow(image)
            ax[j].set_title(f'cluster #{j}', {'fontsize': 35})

    # Draw matrix
    else:
        # Create image
        image = np.zeros((rows, cols))
        image[ys, xs] = 1
        plt.imshow(image)


def divide_matrix(matrix, path, parts=4):
    """
    Divide 1 matrix into parts to make them tangible and write them to files
    :param matrix: df - matrix with intensities
    :param path: str - base path to save parts
    :param parts: int - number of fragments
    :return:
    """
    # Sort index to make slicing possible
    matrix.sort_index(inplace=True)
    
    # Get minimal and maximal xs and ys
    xrange = matrix.index.get_level_values('x').values.min(), matrix.index.get_level_values('x').values.max()
    yrange = matrix.index.get_level_values('y').values.min(), matrix.index.get_level_values('y').values.max()

    # Number of parts on each dimension
    fraction = np.sqrt(parts) + 1

    # Create array with xticks and yticks - 2 parts
    xr = np.linspace(*xrange, fraction).round()
    yr = np.linspace(*yrange, fraction).round()
    part = 0

    # Get 1/4 of matrix by indices and write it to file
    for x, nx in zip(xr, xr[1:]):
        for y, ny in zip(yr, yr[1:]):
            matrix.loc[pd.IndexSlice[:, x:nx, y:ny], :].to_csv(f'{path}_part{part}', sep='\t')
            part += 1


def normalization_tic(matrix):
    """
    Normalize matrix with TIC method - divide everything by matrix sum
    :param matrix: df - dataframe with 1 species
    :return: df - TIC normalized df
    """
    return matrix / matrix.sum().sum()


def load_dirt(path):
    """
    Load json with mz
    :param path: str - path to json with mz
    :return: index - pd index with dirty mz
    """
    with open(path) as file:
        dirt_mz = json.load(file)
    dirt_mz = pd.Float64Index(dirt_mz)
    return dirt_mz


def reindexed_draw_area_clusters(files, n=12, format='png', **kwargs):
    """
    #TODO mb make it more elegant than creating additional functions for reindexed matrices
    Draw k-mean clusterization with number of clusters from 2 to n on 1 plot
    :param files: iterable - collection with full paths to a matrix files
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    for file in files:
        # Load data
        matrix = load_matrix(file, **kwargs)
        print(f'Loaded {file}')

        # Get name of file without extension
        file = file.split('/')[-1].split('.')[0]

        for species in ['h', 'c', 'm']:
            # Get data for 1 species
            subset = matrix.query(f'species == "{species}"')
            print(f'Working with {species} subset')

            # Clustering
            draw_clusters(subset, f'images/{file}/clusters/area/',
                          name=f'kmeans_{species_to_name[species]}_clusters.{format}',
                          n=n, format=format)


def reindexed_draw_peak_clusters(files, n=12, format='png', **kwargs):
    """
    Draw peak k-mean clusterization with number of clusters from 2 to n. There is a picture with all
    :param files: iterable - collection with full paths to a matrix files
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    for file in files:
        # Load data
        matrix = load_matrix(file, **kwargs)
        file = file.split('/')[-1].split('.')[0]
        print(f'Loaded {file}')

        # Create directory
        os.makedirs(f'images/{file}/clusters/peaks', exist_ok=True)

        # For each species clusterize peaks and draw a picture with clusterization
        for species in ['h', 'c', 'm']:
            # Take 1 species
            subset = matrix.query(f'species == "{species}"')
            print(f'Working with {species} subset')

            # Clustering
            clusters = kmeans_clustering(subset.T, n)
            # Write clusterization to file
            with open(f'images/{file}/clusters/peaks/peak_clusters_{species_to_name[species]}.json', 'w') as dest:
                json.dump({cluster: labels.tolist() for cluster, labels in clusters.items()}, dest)

            # Get necessary parameters for plotting
            xs, ys, rows, cols, width_height = light_plot_preparation(subset)

            # For each cluster create superplot
            for i, cluster in clusters.items():
                nrows, ncols, figsize = compute_layout(i, width_height)
                f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
                ax = ax.ravel()

                # Draw each subplot
                for j in range(i):
                    image = np.zeros((rows, cols))
                    image[ys, xs] = subset.loc[:, cluster == j].sum(axis=1)
                    # 2 - minima of cluster number
                    ax[j].imshow(image)
                    ax[j].title.set_text(f'{i} clusters k-means\ncluster #{j}')
                plt.savefig(f'images/{file}/clusters/peaks/kmeans_{species_to_name[species]}_clusters_{i}.{format}',
                            format=format)
    plt.close('all')


def select_peaks(species, peak_number=300, pixel_number=30):
    """
    Select appropriate peaks for correlation computing
    :param species: dict - {'species': df} with species subset of maldi for each species
    :param peak_number: int - number of peaks which should be selected for each species
    :param pixel_number: int - minimal number of pixels with peak to be appropriate
    :return: (dict, dict) - selected peaks for each species with and without counting blank pixels
    """
    # Prepare containers
    max_intensity_peaks = {}
    max_intensity_peaks_na = {}

    # Get peaks with top mean intensity and number of pixels with value greater than threshold
    # Compute mean with and without considering blank pixels
    for sp, df in species.items():
        dfn = df.copy()

        max_intensity_peaks[sp] = df.loc[:, (df != 0).sum() > pixel_number].mean().nlargest(peak_number).index
        dfn[dfn == 0] = np.nan
        max_intensity_peaks_na[sp] = dfn.loc[:, (dfn != 0).sum() > pixel_number].mean().nlargest(peak_number).index

    return max_intensity_peaks, max_intensity_peaks_na


def many_correlations(lcms, maldi2, span=(-100, 100, 5), method='absolute'):
    """
    Compute correlations for different recalibration
    :param lcms: df - LC dataset
    :param maldi: df - MALDI dataset
    :param span: tuple - (start, stop, step) for recalibration
    :param method: str - recalibration method, absolute by default
    :return: df - correlations for each recalibration value
    """
    colnames = ['human_chimp_ratio_pearson',
                'human_chimp_ratio_spearman',
                'human_macaque_ratio_pearson',
                'human_macaque_ratio_spearman',
                'peak_number']

    # Compute correlations for different recalibrations
    correlations = recalibrate_align(lcms, maldi2, span, method)

    # Create df from correlation data
    df = pd.DataFrame(correlations, index=np.arange(*span), columns=colnames)
    return df


def plot_correlations(correlations, cutoff=30):
    """
    Visualize df with correlations on different recalibrations in form of histogram
    :param correlations: df - correlations on different offsets
    :param cutoff: int - number of peaks which are considered as a threshold starting from which correlation is
    informative
    :return:
    """
    # Compute some constants
    max_peaks = correlations["peak_number"].max()
    step = correlations.index[1] - correlations.index[0]
    span = correlations.index[0], correlations.index[-1] + step, step

    # Nullify correlations with small aligned peaks
    correlations.loc[correlations['peak_number'] < cutoff, ['human_chimp_ratio_pearson', 'human_chimp_ratio_spearman',
                                            'human_macaque_ratio_pearson', 'human_macaque_ratio_spearman']] = 0

    # Draw plot
    # Correlations
    correlations.drop(columns=['peak_number']).plot(figsize=(12, 8))
    # Aligned peak fraction
    plt.plot(correlations['peak_number'] / max_peaks, linestyle='--', label=f'peak_number_fraction, max {max_peaks}')
    # Horizontal edge of minimal appropriate peak number
    plt.hlines(cutoff / max_peaks, *span[:2], '#642e55', label=f'{cutoff} threshold')
    plt.legend()


# Example of usage
if __name__ == '__main__':
    # Load matrix
    matrix = load_matrix('~/data/maldi/matrices/feature_matrix_new_spectra_06_09_2018_p1.tsv', sep='\t')

    # Zero pixels
    zeros(matrix)
    # Multiindex
    reindexing(matrix)
    # # of duplicates
    duplications(matrix)

    # Take only human
    human_data = matrix.query('species == "h"')

    # Pixel coordinates
    xs, ys = get_coords(human_data)
    # Extract [740, 741) ions
    ions740 = human_data.filter(regex=r'^740.*').columns

    # Draw picture with all 740 ions
    draw_panel(matrix, xs, ys, ions740, width_height=(6, 4), save=False, path='img', format='png')

    # Statistics of pixel failure
    print(*pixel_stats(human_data, 20), sep='\n\n')

    # Get mz_intensities
    mz_intensity = matrix_to_mz_intensities(human_data.sum())

    # Merge similar peaks
    mz_intensity_merged = merge(mz_intensity, 5)

    # Plot intensities of all mz
    mz_intensity_plot(mz_intensity_merged)
    # Plot intensities of range of mz
    mz_intensity_plot(mz_intensity_merged, (700, 715))

