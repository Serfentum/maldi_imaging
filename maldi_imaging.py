import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, MiniBatchKMeans


def load_matrix(path, sep='\t'):
    """
    Loading matrix obtained after MALDIQuant library processing of .RAW files
    :param path: str - path to the matrix
    :param sep: str - field separator in file
    :return: df - pandas df with data
    """
    matrix = pd.read_csv(path, sep=sep)
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
    # No totally blank pixels if pass
    assert not is_blank_pixel.any(), 'There are some blank pixels'

    # Number of pixels which have 0 intensity for ions
    mz_zero_num = elem_is_zero.sum()

    # Which ions have intensity equal to 0 in all pixels
    is_blank_mz = (mz_zero_num == matrix.shape[0])
    # No totally blank mz if it pass
    assert not is_blank_mz.any(), print('There are some blank mz')


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
            merged[-1], intensities[-1] = merge(merged[-1], mz, intensities[-1], current_int)
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
    lcms = pd.read_csv(lcms, sep='\t', index_col=0)
    matrix = pd.read_csv(matrix, sep='\t')

    # Multiindex
    reindexing(matrix)
    return lcms, matrix


def naive_align_peaks(lcms, matrix, threshold=5):
    # Convert ppm to fraction
    threshold *= 1e-6

    # Get relative differences between each ion in LC and MALDI
    lc_diffs = {lc: {(m, difference(lc, m)) for m in matrix.columns} for lc in lcms.mz}
    # Get only mz with minimal ppm difference - will contains only 1 variant for ion
    lc_diffs = {lc: min(diffs, key=lambda x: x[1]) for lc, diffs in lc_diffs.items()}
    # Filter out peaks with relative difference > than 5ppm
    lc_diffs = {lc: diff for lc, diff in lc_diffs.items() if diff[1] <= threshold}

    # Dictionary with correspondance between MALDI and LC aligned peaks
    renaming = {m: lc for lc, (m, _) in lc_diffs.items()}
    print(f'Number of aligned peaks is {len(renaming)}')

    # Make peak mz consistent in MALDI and LC
    aligned_matrix = matrix.rename(columns=renaming)
    aligned_matrix = aligned_matrix[list(sorted(renaming.values()))]
    return aligned_matrix, renaming


def get_ratio(intensities1, intensities2):
    """
    Get ratio of corresponding intensities from intensities1 and intensities2
    :param intensities1: iterable - collection with intensities in a form of series or scalar
    :param intensities2: iterable - collection with intensities in a form of series or scalar
    :return: list - list with ratios of corresponding intensities
    """
    ratios = []

    # Iterate over each intensity and divide 1st on the 2nd
    for ints1, ints2 in zip(intensities1, intensities2):
        ratios.append(ints1 / ints2)
    return ratios


def get_correlations(intensities1, intensities2):
    """
    Compute correlations of corresponding intensities from intensities1 and intensities2
    :param intensities1: series - series with intensities
    :param intensities2: series - series with intensities
    :return: list - list with correlations between pairs of intensity series
    """
    corrs = []
    for ints1, ints2 in zip(intensities1, intensities2):
        corrs.append(ints1.corr(ints2))
    return corrs


def subsetting_mean(lcms, matrix, renaming):
    # todo generalize
    """
    Take groups from LC and MALDI and return mean for each peak in each group
    :param matrix: df - df with all data
    :return: (series, series, series, series, series, series) - tuple of series with mean for each group
    """
    # Use mz as an index
    lcms.set_index('mz', inplace=True)

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
          '#008080': [0, 128, 128]}

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
               '#1E90FF', '#4B0082', '#008080']}
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
    # Write clusterization to file
    with open(f'{path}/area_clusters_{species_to_name[species]}.json', 'w') as file:
        json.dump({cluster: labels.tolist() for cluster, labels in clusters.items()}, file)

    # Preparations to plotting
    xs, ys, nrows, ncols, figsize, rows, cols = _auxiliary(data, n)
    # Plotting
    _draw_clusters(clusters, xs, ys, nrows, ncols, figsize, rows, cols)

    # Save figure and close everything
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/{name}.{format}', format=format)
    plt.close()


# Mapping from letter to species
species_to_name = {'h': 'human',
                   'c': 'chimp',
                   'm': 'macaque'}


def draw_area_clusters(files, n=12, format='png'):
    """
    Draw k-mean clusterization with number of clusters from 2 to n on 1 plot
    :param files: iterable - collection with full paths to a matrix files
    :param n: int - maximum number of clusters, 12 by default
    :param format: str - format of figure, png by default
    :return:
    """
    for file in files:
        # Load data
        matrix = load_matrix(f'{file}')
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
            with open(f'images/{file}/clusters/peaks/peak_clusters_{species_to_name[species]}.json', 'w') as file:
                json.dump({cluster: labels.tolist() for cluster, labels in clusters.items()}, file)

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
    with open(path) as file:
        clusters = json.load(file)

    # Convert cluster number to int and lists to np.array
    clusters = {int(cluster): np.array(labels) for cluster, labels in clusters.items()}
    return clusters


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

