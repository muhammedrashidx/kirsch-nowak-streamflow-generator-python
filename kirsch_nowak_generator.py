"""
Kirsch–Nowak Streamflow Generator — Python implementation

This implementation is based on the stationary synthetic streamflow
generation methodology and original MATLAB implementation developed by
Matteo Giuliani, Jon Herman, and Julianne Quinn.

Original repository:
https://github.com/julianneq/Kirsch-Nowak_Streamflow_Generator

This Python implementation was developed to provide a portable Python
implementation with MATLAB-compatible stochastic behavior, including
reproducible random-number generation and realization ordering.

The original methodology is described in:
- Kirsch et al. (2013)
- Nowak et al. (2010)

Author of this Python implementation:
Muhammed Rashid
2026
"""

"""
===============================================================
KIRSCH - NOWAK SYNTHETIC STREAMFLOW GENERATOR
===============================================================

Purpose
-------
Generate synthetic daily streamflow in two stages:

    KIRSCH
        Historical daily streamflow
                ↓
        Monthly streamflow
                ↓
        Log transformation
                ↓
        Monthly standardization
                ↓
        Bootstrap resampling
                ↓
        Historical correlation matrix
                ↓
        Cholesky decomposition
                ↓
        Correlated synthetic monthly streamflow
                ↓
        Back transformation

    NOWAK
        Synthetic monthly streamflow
                ↓
        Find similar historical months
                ↓
        K-nearest neighbours
                ↓
        Randomly select one historical pattern
                ↓
        Extract daily proportions
                ↓
        Scale proportions to synthetic monthly flow
                ↓
        Synthetic daily streamflow


References
----------

Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013).
Evaluating the impact of alternative hydro-climate scenarios
on transfer agreements: Practical improvement for generating
synthetic streamflows.
Journal of Water Resources Planning and Management, 139(4),
396-406.

Nowak, K., Prairie, J., Rajagopalan, B., & Lall, U. (2010).
A nonparametric stochastic approach for multisite disaggregation
of annual to daily streamflow.
Water Resources Research, 46, W08529.

===============================================================
"""

import csv

import numpy as np


# ===============================================================
# SETTINGS
# ===============================================================

DAYS_IN_MONTH = np.array([
    31, 28, 31, 30, 31, 30,
    31, 31, 30, 31, 30, 31
])

MONTH_NAMES = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]


# ===============================================================
# PART 1
# DAILY DATA → MONTHLY DATA
# ===============================================================

def daily_to_monthly(daily_flow):
    """
    Convert daily streamflow to monthly totals.

    Parameters
    ----------
    daily_flow : ndarray
        Shape:
            (years * 365, sites)

    Returns
    -------
    monthly_flow : ndarray
        Shape:
            (years, 12, sites)

    Each month is obtained by summing the daily flows
    belonging to that month.

    Leap days should already have been removed.
    """

    daily_flow = np.asarray(daily_flow, dtype=float)

    number_of_days, number_of_sites = daily_flow.shape

    if number_of_days % 365 != 0:
        raise ValueError(
            "Daily data must contain complete 365-day years."
        )

    number_of_years = number_of_days // 365

    monthly_flow = np.zeros(
        (number_of_years, 12, number_of_sites)
    )

    for year in range(number_of_years):

        day_position = year * 365

        for month in range(12):

            days_this_month = DAYS_IN_MONTH[month]

            start = day_position
            end = start + days_this_month

            monthly_flow[year, month, :] = np.sum(
                daily_flow[start:end, :],
                axis=0
            )

            day_position = end

    return monthly_flow


# ===============================================================
# PART 2
# LOG TRANSFORMATION + STANDARDIZATION
# ===============================================================

def standardize_monthly_flow(monthly_flow):
    """
    Log-transform and standardize each month separately.

    For every month:

        log(Q)

    followed by:

        Z = [log(Q) - mean] / standard_deviation

    Returns
    -------
    standardized_flow
        Standardized historical monthly values.

    monthly_log_mean
        Mean log-flow for each month.

    monthly_log_std
        Standard deviation of log-flow for each month.
    """

    monthly_flow = np.asarray(
        monthly_flow,
        dtype=float
    )

    if np.any(monthly_flow <= 0):
        raise ValueError(
            "All monthly flows must be positive "
            "because logarithms are used."
        )

    # -----------------------------------------------------------
    # Log transformation
    # -----------------------------------------------------------

    log_flow = np.log(monthly_flow)

    # -----------------------------------------------------------
    # Mean and standard deviation for each calendar month
    #
    # Example:
    #
    # monthly_log_mean[0] = mean January log-flow
    # monthly_log_mean[1] = mean February log-flow
    # ...
    # -----------------------------------------------------------

    monthly_log_mean = np.mean(
        log_flow,
        axis=0
    )

    monthly_log_std = np.std(
        log_flow,
        axis=0,
        ddof=1
    )

    # -----------------------------------------------------------
    # Standardization
    # -----------------------------------------------------------

    standardized_flow = (
        log_flow - monthly_log_mean
    ) / monthly_log_std

    return (
        standardized_flow,
        monthly_log_mean,
        monthly_log_std
    )


# ===============================================================
# PART 3
# CORRELATION MATRIX + CHOLESKY
# ===============================================================

def calculate_cholesky_factor(standardized_flow):
    """
    Calculate the historical correlation matrix and
    its Cholesky factor.

    Parameters
    ----------
    standardized_flow : ndarray
        Shape:
            (years, 12)

    Returns
    -------
    correlation_matrix : ndarray
        12 x 12 historical correlation matrix.

    cholesky_factor : ndarray
        Upper-triangular Cholesky factor.

    Mathematical relationship
    -------------------------
    R = U.T @ U

    where:

        R = historical correlation matrix
        U = upper-triangular Cholesky factor

    Synthetic correlated values are then obtained using:

        Z_correlated = Z_uncorrelated @ U
    """

    standardized_flow = np.asarray(
        standardized_flow,
        dtype=float
    )

    # -----------------------------------------------------------
    # Correlation between the 12 monthly columns
    # -----------------------------------------------------------

    correlation_matrix = np.corrcoef(
        standardized_flow,
        rowvar=False
    )

    # Remove very small numerical asymmetry.
    correlation_matrix = (
        correlation_matrix
        + correlation_matrix.T
    ) / 2.0

    # -----------------------------------------------------------
    # Cholesky decomposition
    #
    # numpy gives:
    #
    #     R = L @ L.T
    #
    # We use:
    #
    #     U = L.T
    #
    # so that:
    #
    #     R = U.T @ U
    #
    # This is the upper-triangular form used in this generator.
    # -----------------------------------------------------------

    while True:
        try:
            lower_factor = np.linalg.cholesky(
                correlation_matrix
            )
            break
        except np.linalg.LinAlgError:
            eigenvalues = np.linalg.eigvals(
                correlation_matrix
            )
            k = min(
                np.min(np.real(eigenvalues)),
                -np.finfo(float).eps
            )
            correlation_matrix = (
                correlation_matrix
                - k * np.eye(correlation_matrix.shape[0])
            )
            correlation_matrix = (
                correlation_matrix / correlation_matrix[0, 0]
            )

    upper_factor = lower_factor.T

    return (
        correlation_matrix,
        upper_factor
    )


# ===============================================================
# PART 4
# BOOTSTRAP
# ===============================================================

def bootstrap_monthly_values(
    standardized_historical,
    number_of_synthetic_years,
    rng
):
    """
    Generate bootstrap samples of standardized monthly flow.

    For every synthetic year and every month:

        randomly select one historical year

    Example:

        random_indices[5, 2] = 17

    means:

        synthetic year 5
        March
        ← March from historical year 17

    The selection is independent for each month.

    Returns
    -------
    random_indices
        Historical year selected for each
        synthetic year and month.

    uncorrelated_synthetic
        Bootstrap synthetic standardized flow.
    """

    number_of_historical_years = (
        standardized_historical.shape[0]
    )

    # -----------------------------------------------------------
    # One additional internal year is generated because the
    # shifted 12-month construction needs it.
    # -----------------------------------------------------------

    internal_years = (
        number_of_synthetic_years + 1
    )

    # -----------------------------------------------------------
    # Random historical-year indices
    #
    # -----------------------------------------------------------

    r_floats = rng.rand(12, internal_years).T
    random_indices = np.floor(
        r_floats * number_of_historical_years
    ).astype(int)

    # -----------------------------------------------------------
    # Construct bootstrap synthetic matrix
    # -----------------------------------------------------------

    uncorrelated_synthetic = np.zeros(
        (internal_years, 12)
    )

    for month in range(12):

        historical_rows = (
            random_indices[:, month]
        )

        uncorrelated_synthetic[:, month] = (
            standardized_historical[
                historical_rows,
                month
            ]
        )

    return (
        random_indices,
        uncorrelated_synthetic
    )


# ===============================================================
# PART 5
# CREATE SHIFTED 12-MONTH DATA
# ===============================================================

def create_shifted_data(data):
    """
    Shift the 12-month sequence by six months.

    Original order:

        Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec

    Shifted order:

        Jul Aug Sep Oct Nov Dec Jan Feb Mar Apr May Jun

    The purpose is to create a continuous 12-month window
    across the December → January boundary.

    The result is still a 12-column matrix.
    Only the order of the observations has changed.
    """

    data = np.asarray(
        data,
        dtype=float
    )

    # -----------------------------------------------------------
    # Convert:
    #
    # year 1: Jan...Dec
    # year 2: Jan...Dec
    #
    # into one continuous sequence.
    # -----------------------------------------------------------

    continuous_sequence = data.reshape(-1)

    # -----------------------------------------------------------
    # Remove six values at the beginning and six at the end.
    #
    # This creates complete shifted 12-month windows.
    # -----------------------------------------------------------

    shifted_sequence = (
        continuous_sequence[6:-6]
    )

    # -----------------------------------------------------------
    # Rebuild as 12-month rows.
    # -----------------------------------------------------------

    shifted_data = (
        shifted_sequence.reshape(-1, 12)
    )

    return shifted_data


# ===============================================================
# PART 6
# KIRSCH MONTHLY GENERATOR
# ===============================================================

def generate_kirsch_monthly(
    historical_monthly,
    number_of_synthetic_years,
    rng
):
    """
    Generate synthetic monthly streamflow.

    Steps
    -----

    1. Log-transform historical flow.
    2. Standardize each month.
    3. Calculate normal historical correlation.
    4. Calculate shifted historical correlation.
    5. Cholesky-decompose both correlation matrices.
    6. Bootstrap historical standardized values.
    7. Impose normal correlation.
    8. Shift the bootstrap sequence.
    9. Impose shifted correlation.
    10. Combine the two correlated sequences.
    11. Reverse standardization.
    12. Reverse the logarithm.

    Returns
    -------
    synthetic_monthly
        Shape:
            (number_of_synthetic_years, 12)

    diagnostics
        Intermediate objects for checking the calculation.
    """

    historical_monthly = np.asarray(
        historical_monthly,
        dtype=float
    )

    number_of_historical_years = (
        historical_monthly.shape[0]
    )

    if historical_monthly.shape[1] != 12:
        raise ValueError(
            "Historical monthly data must contain 12 months."
        )

    # ===========================================================
    # STEP 1
    # Log transformation + standardization
    # ===========================================================

    (
        standardized_historical,
        monthly_log_mean,
        monthly_log_std
    ) = standardize_monthly_flow(
        historical_monthly
    )

    # ===========================================================
    # STEP 2
    # Normal historical correlation
    # ===========================================================

    (
        normal_correlation,
        normal_cholesky
    ) = calculate_cholesky_factor(
        standardized_historical
    )

    # ===========================================================
    # STEP 3
    # Shifted historical data
    # ===========================================================

    shifted_historical = (
        create_shifted_data(
            standardized_historical
        )
    )

    # ===========================================================
    # STEP 4
    # Shifted historical correlation
    # ===========================================================

    (
        shifted_correlation,
        shifted_cholesky
    ) = calculate_cholesky_factor(
        shifted_historical[
            :number_of_historical_years - 1,
            :
        ]
    )

    # ===========================================================
    # STEP 5
    # Bootstrap
    # ===========================================================

    # -----------------------------------------------------------
    # 1. Bootstrap historical standardized matrix
    # -----------------------------------------------------------

    (
        random_indices,
        uncorrelated_synthetic
    ) = bootstrap_monthly_values(
        standardized_historical,
        number_of_synthetic_years,
        rng
    )

    # ===========================================================
    # STEP 6
    # Apply normal correlation
    #
    #     Z_normal = Z_uncorrelated @ U
    #
    # ===========================================================

    correlated_normal = (
        uncorrelated_synthetic
        @ normal_cholesky
    )

    # ===========================================================
    # STEP 7
    # Shift synthetic bootstrap sequence
    # ===========================================================

    uncorrelated_shifted = (
        create_shifted_data(
            uncorrelated_synthetic
        )
    )

    # ===========================================================
    # STEP 8
    # Apply shifted correlation
    #
    #     Z_shifted = Z_uncorrelated_shifted @ U_shifted
    #
    # ===========================================================

    correlated_shifted = (
        uncorrelated_shifted
        @ shifted_cholesky
    )

    # ===========================================================
    # STEP 9
    # Combine the two correlated sequences
    #
    # Shifted sequence:
    #
    #     Jul Aug Sep Oct Nov Dec Jan Feb Mar Apr May Jun
    #
    # Therefore:
    #
    #     columns 7-12 = Jan Feb Mar Apr May Jun
    #
    # Normal sequence:
    #
    #     Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
    #
    # Therefore:
    #
    #     columns 7-12 = Jul Aug Sep Oct Nov Dec
    # ===========================================================

    final_standardized = np.zeros(
        (number_of_synthetic_years, 12)
    )

    # January to June
    final_standardized[:, 0:6] = (
        correlated_shifted[:, 6:12]
    )

    # July to December
    final_standardized[:, 6:12] = (
        correlated_normal[1:, 6:12]
    )

    # ===========================================================
    # STEP 10
    # Reverse standardization
    #
    #     log(Q) = Z * standard_deviation + mean
    # ===========================================================

    synthetic_log_flow = (
        final_standardized
        * monthly_log_std
        + monthly_log_mean
    )

    # ===========================================================
    # STEP 11
    # Reverse log transformation
    #
    #     Q = exp(log(Q))
    # ===========================================================

    synthetic_monthly = np.exp(
        synthetic_log_flow
    )

    # ===========================================================
    # Diagnostics
    # ===========================================================

    diagnostics = {

        "historical_standardized":
            standardized_historical,

        "monthly_log_mean":
            monthly_log_mean,

        "monthly_log_std":
            monthly_log_std,

        "normal_correlation":
            normal_correlation,

        "normal_cholesky":
            normal_cholesky,

        "shifted_historical":
            shifted_historical,

        "shifted_correlation":
            shifted_correlation,

        "shifted_cholesky":
            shifted_cholesky,

        "random_indices":
            random_indices,

        "uncorrelated_synthetic":
            uncorrelated_synthetic,

        "correlated_normal":
            correlated_normal,

        "uncorrelated_shifted":
            uncorrelated_shifted,

        "correlated_shifted":
            correlated_shifted,

        "final_standardized":
            final_standardized,

        "synthetic_log_flow":
            synthetic_log_flow
    }

    return (
        synthetic_monthly,
        diagnostics
    )


# ===============================================================
# PART 7
# NOWAK: CREATE HISTORICAL MONTHLY PATTERN LIBRARY
# ===============================================================

def build_historical_monthly_library(
    historical_daily
):
    """
    Build the historical monthly patterns used for KNN.

    For each calendar month, historical monthly totals are
    calculated using 15 shifted versions corresponding to
    approximately +/- 7 days around the normal month boundary.

    Returns
    -------
    monthly_totals
        List of 12 arrays.

        monthly_totals[month]
        contains historical monthly totals for that month.

    pattern_information
        Information identifying:

            historical year
            shift number

        for every stored pattern.
    """

    historical_daily = np.asarray(
        historical_daily,
        dtype=float
    )

    number_of_days, number_of_sites = (
        historical_daily.shape
    )

    number_of_years = (
        number_of_days // 365
    )

    # -----------------------------------------------------------
    # Add seven days before and eight days after the record.
    #
    # This allows monthly windows to move around the
    # calendar-month boundary.
    # -----------------------------------------------------------

    extended_daily = np.vstack([
        historical_daily[-8:, :],
        historical_daily,
        historical_daily[:8, :]
    ])

    monthly_totals = []
    pattern_information = []

    # ===========================================================
    # For each calendar month
    # ===========================================================

    for month in range(12):

        month_values = []
        month_information = []

        # -------------------------------------------------------
        # 15 possible shifts
        #
        # shift = 1,...,15
        #
        # corresponding approximately to:
        #
        # -7, -6, ..., 0, ..., +6, +7 days
        # -------------------------------------------------------

        for shift in range(1, 16):

            start = shift - 1

            shifted_daily = (
                extended_daily[
                    start:start + number_of_days,
                    :
                ]
            )

            shifted_monthly = (
                daily_to_monthly(
                    shifted_daily
                )
            )

            values = (
                shifted_monthly[
                    :,
                    month,
                    0
                ]
            )

            year_indices = np.arange(
                number_of_years
            )

            # ---------------------------------------------------
            # Remove incomplete boundary years.
            # ---------------------------------------------------

            if month == 0 and shift < 8:

                values = values[1:]
                year_indices = year_indices[1:]

            if month == 11 and shift > 8:

                values = values[:-1]
                year_indices = year_indices[:-1]

            month_values.append(
                values
            )

            information = np.column_stack([
                year_indices,
                np.full(
                    len(year_indices),
                    shift
                )
            ])

            month_information.append(
                information
            )

        monthly_totals.append(
            np.concatenate(month_values)
        )

        pattern_information.append(
            np.vstack(month_information)
        )

    return (
        monthly_totals,
        pattern_information
    )


# ===============================================================
# PART 8
# NOWAK: FIND K NEAREST NEIGHBOURS
# ===============================================================

def find_nearest_neighbors(
    synthetic_monthly_value,
    historical_monthly_values,
    number_of_neighbors=None
):
    """
    Find K historical monthly totals closest to the
    synthetic monthly total.

    Distance:

        distance =
        (historical_value - synthetic_value)^2

    K:

        K = round(sqrt(N))

    when K is not explicitly supplied.

    Weights:

        w_i = (1/i) / sum(1/i)

    where i is the neighbour rank.
    """

    historical_monthly_values = np.asarray(
        historical_monthly_values,
        dtype=float
    )

    number_of_patterns = (
        len(historical_monthly_values)
    )

    if number_of_neighbors is None:

        number_of_neighbors = int(
            np.round(
                np.sqrt(number_of_patterns)
            )
        )

    number_of_neighbors = min(
        number_of_neighbors,
        number_of_patterns
    )

    # -----------------------------------------------------------
    # Squared distance
    # -----------------------------------------------------------

    distances = (
        historical_monthly_values
        - synthetic_monthly_value
    ) ** 2

    # -----------------------------------------------------------
    # Sort by distance
    # -----------------------------------------------------------

    sorted_indices = np.argsort(
        distances,
        kind="stable"
    )

    nearest_indices = (
        sorted_indices[
            :number_of_neighbors
        ]
    )

    # -----------------------------------------------------------
    # Inverse-rank weights
    # -----------------------------------------------------------

    ranks = np.arange(
        1,
        number_of_neighbors + 1
    )

    weights = 1.0 / ranks

    weights = (
        weights / np.sum(weights)
    )

    return (
        nearest_indices,
        weights
    )


# ===============================================================
# PART 9
# NOWAK: RANDOMLY SELECT ONE NEIGHBOUR
# ===============================================================

def select_neighbor(
    nearest_indices,
    weights,
    rng
):
    """
    Randomly select one of the K nearest neighbours
    according to its probability weight.
    """

    r = rng.rand()
    cumulative_weights = np.cumsum(weights)
    selected_position = np.searchsorted(cumulative_weights, r)

    selected_index = (
        nearest_indices[
            selected_position
        ]
    )

    return selected_index


# ===============================================================
# PART 10
# NOWAK: EXTRACT DAILY PROPORTIONS
# ===============================================================

def get_daily_proportions(
    historical_daily,
    historical_year,
    shift,
    month
):
    """
    Extract the daily flow proportions from the selected
    historical month.

    If the historical monthly flow is:

        Q_month

    and the daily values are:

        q_1, q_2, ..., q_n

    then:

        p_i = q_i / Q_month

    so:

        sum(p_i) = 1

    The synthetic daily values are later:

        q_synthetic_i
        =
        p_i * Q_synthetic_month
    """

    historical_daily = np.asarray(
        historical_daily,
        dtype=float
    )

    number_of_days = (
        historical_daily.shape[0]
    )

    # -----------------------------------------------------------
    # Extend the historical record by seven days at the
    # beginning and eight days at the end.
    # -----------------------------------------------------------

    extended_daily = np.vstack([
        historical_daily[-8:, :],
        historical_daily,
        historical_daily[:8, :]
    ])

    # -----------------------------------------------------------
    # Apply the selected shift.
    # -----------------------------------------------------------

    shift_start = shift - 1

    shifted_daily = (
        extended_daily[
            shift_start:
            shift_start + number_of_days,
            :
        ]
    )

    # -----------------------------------------------------------
    # Find the target month.
    # -----------------------------------------------------------

    year_start = (
        historical_year * 365
    )

    month_start = (
        year_start
        + np.sum(
            DAYS_IN_MONTH[:month]
        )
    )

    month_end = (
        month_start
        + DAYS_IN_MONTH[month]
    )

    daily_values = (
        shifted_daily[
            month_start:month_end,
            :
        ]
    )

    # -----------------------------------------------------------
    # Convert daily values to proportions.
    # -----------------------------------------------------------

    monthly_total = np.sum(
        daily_values,
        axis=0
    )

    if np.any(monthly_total <= 0):

        raise ValueError(
            "Selected historical month contains "
            "zero or negative total flow."
        )

    daily_proportions = (
        daily_values
        / monthly_total
    )

    return daily_proportions


# ===============================================================
# PART 11
# NOWAK: MONTHLY → DAILY
# ===============================================================

def nowak_disaggregation(
    synthetic_monthly,
    historical_daily,
    rng,
    number_of_neighbors=None
):
    """
    Convert synthetic monthly streamflow into synthetic
    daily streamflow.

    Parameters
    ----------
    synthetic_monthly : ndarray
        Shape:

            (synthetic_years, 12, sites)

    historical_daily : ndarray
        Shape:

            (historical_years * 365, sites)

    Returns
    -------
    synthetic_daily : ndarray
        Shape:

            (synthetic_years * 365, sites)
    """

    synthetic_monthly = np.asarray(
        synthetic_monthly,
        dtype=float
    )

    historical_daily = np.asarray(
        historical_daily,
        dtype=float
    )

    number_of_synthetic_years = (
        synthetic_monthly.shape[0]
    )

    number_of_sites = (
        synthetic_monthly.shape[2]
    )

    # -----------------------------------------------------------
    # Build historical monthly library.
    # -----------------------------------------------------------

    (
        historical_monthly_library,
        pattern_information
    ) = build_historical_monthly_library(
        historical_daily
    )

    # -----------------------------------------------------------
    # Pre-allocate synthetic daily.
    # -----------------------------------------------------------

    synthetic_daily = np.zeros(
        (
            number_of_synthetic_years * 365,
            number_of_sites
        )
    )

    # ===========================================================
    # Loop through synthetic years
    # ===========================================================

    for year in range(
        number_of_synthetic_years
    ):

        # =======================================================
        # Loop through months
        # =======================================================

        for month in range(12):

            # ---------------------------------------------------
            # Synthetic monthly flow.
            #
            # The neighbour search uses the first site.
            # ---------------------------------------------------

            target_monthly_flow = (
                synthetic_monthly[
                    year,
                    month,
                    0
                ]
            )

            # ---------------------------------------------------
            # Find K nearest historical monthly totals.
            # ---------------------------------------------------

            (
                nearest_indices,
                weights
            ) = find_nearest_neighbors(
                target_monthly_flow,
                historical_monthly_library[
                    month
                ],
                number_of_neighbors
            )

            # ---------------------------------------------------
            # Randomly select one historical pattern.
            # ---------------------------------------------------

            selected_pattern = (
                select_neighbor(
                    nearest_indices,
                    weights,
                    rng
                )
            )

            # ---------------------------------------------------
            # Identify the historical year and shift
            # corresponding to the selected pattern.
            # ---------------------------------------------------

            selected_year = int(
                pattern_information[
                    month
                ][
                    selected_pattern,
                    0
                ]
            )

            selected_shift = int(
                pattern_information[
                    month
                ][
                    selected_pattern,
                    1
                ]
            )

            # ---------------------------------------------------
            # Extract historical daily proportions.
            # ---------------------------------------------------

            daily_proportions = (
                get_daily_proportions(
                    historical_daily,
                    selected_year,
                    selected_shift,
                    month
                )
            )

            # ---------------------------------------------------
            # Scale proportions by synthetic monthly flow.
            #
            # Each site receives its own synthetic monthly total.
            # ---------------------------------------------------

            synthetic_monthly_vector = (
                synthetic_monthly[
                    year,
                    month,
                    :
                ]
            )

            synthetic_daily_values = (
                daily_proportions
                * synthetic_monthly_vector
            )

            # ---------------------------------------------------
            # Location of this month in the output.
            # ---------------------------------------------------

            year_start = (
                year * 365
            )

            month_start = (
                year_start
                + np.sum(
                    DAYS_IN_MONTH[:month]
                )
            )

            month_end = (
                month_start
                + DAYS_IN_MONTH[month]
            )

            synthetic_daily[
                month_start:month_end,
                :
            ] = synthetic_daily_values

    return synthetic_daily


# ===============================================================
# PART 12
# COMPLETE KIRSCH + NOWAK GENERATOR
# ===============================================================

def generate_synthetic_streamflow(
    historical_daily,
    number_of_realizations,
    number_of_years,
    random_seed=None
):
    """
    Complete synthetic streamflow generator.

    Input
    -----
    historical_daily:
        Historical daily streamflow.

        Shape:

            (historical_years * 365, sites)

    Output
    ------
    synthetic_daily:
        Shape:

            (realizations,
             synthetic_years * 365,
             sites)

    synthetic_monthly:
        Shape:

            (realizations,
             synthetic_years * 12,
             sites)
    """

    historical_daily = np.asarray(
        historical_daily,
        dtype=float
    )

    number_of_sites = (
        historical_daily.shape[1]
    )

    # ===========================================================
    # Historical daily → monthly
    # ===========================================================

    historical_monthly = (
        daily_to_monthly(
            historical_daily
        )
    )

    number_of_historical_years = (
        historical_monthly.shape[0]
    )

    print(
        f"Historical years: "
        f"{number_of_historical_years}"
    )

    print(
        f"Number of sites: "
        f"{number_of_sites}"
    )

    # ===========================================================
    # Output arrays
    # ===========================================================

    synthetic_daily_all = np.zeros(
        (
            number_of_realizations,
            number_of_years * 365,
            number_of_sites
        )
    )

    synthetic_monthly_all = np.zeros(
        (
            number_of_realizations,
            number_of_years * 12,
            number_of_sites
        )
    )

    rng = np.random.RandomState(
        random_seed
    )

    # ===========================================================
    # Generate each realization
    # ===========================================================

    synthetic_monthly_3d_all_realizations = []

    for realization in range(
        number_of_realizations
    ):

        print(
            f"Generating monthly realization "
            f"{realization + 1}/"
            f"{number_of_realizations}"
        )

        # -------------------------------------------------------
        # KIRSCH
        # -------------------------------------------------------

        monthly_for_all_sites = []

        for site in range(
            number_of_sites
        ):

            historical_monthly_site = (
                historical_monthly[
                    :,
                    :,
                    site
                ]
            )

            (
                synthetic_monthly_site,
                diagnostics
            ) = generate_kirsch_monthly(
                historical_monthly_site,
                number_of_years,
                rng=rng
            )

            monthly_for_all_sites.append(
                synthetic_monthly_site
            )

        synthetic_monthly_3d = np.stack(
            monthly_for_all_sites,
            axis=2
        )

        # Save monthly output.
        synthetic_monthly_all[
            realization
        ] = synthetic_monthly_3d.reshape(
            number_of_years * 12,
            number_of_sites
        )
        
        synthetic_monthly_3d_all_realizations.append(synthetic_monthly_3d)

    # -------------------------------------------------------
    # NOWAK
    # -------------------------------------------------------
    
    for realization in range(
        number_of_realizations
    ):

        print(
            f"Generating daily realization "
            f"{realization + 1}/"
            f"{number_of_realizations}"
        )
        
        synthetic_monthly_3d = synthetic_monthly_3d_all_realizations[realization]

        synthetic_daily = (
            nowak_disaggregation(
                synthetic_monthly_3d,
                historical_daily,
                rng=rng
            )
        )

        synthetic_daily_all[
            realization
        ] = synthetic_daily

    return (
        synthetic_daily_all,
        synthetic_monthly_all
    )


def save_synthetic_csv(filename, synthetic_flow):
    """Save synthetic flow with realization as rows and time as columns as CSV."""

    synthetic_flow = np.asarray(
        synthetic_flow,
        dtype=float
    )

    with open(filename, "w", newline="") as output_file:

        writer = csv.writer(output_file)

        for realization in range(synthetic_flow.shape[0]):
            row = [
                f"{value:.18g}"
                for value in synthetic_flow[realization].flatten()
            ]
            writer.writerow(row)


# ===============================================================
# PART 13
# MAIN PROGRAM
# ===============================================================

if __name__ == "__main__":

    # -----------------------------------------------------------
    # USER SETTINGS
    # -----------------------------------------------------------

    historical_file = "Qdaily.csv"

    number_of_realizations = 1000

    number_of_synthetic_years = 33

    random_seed = 12345

    # -----------------------------------------------------------
    # READ HISTORICAL DAILY DATA
    #
    # CSV format:
    #
    # rows    = daily observations
    # columns = streamflow sites
    #
    # Example for one site:
    #
    # 33 years × 365 days = 12045 rows
    # -----------------------------------------------------------

    historical_daily = np.loadtxt(
        historical_file,
        delimiter=","
    )

    # Make one-site data two-dimensional.
    if historical_daily.ndim == 1:

        historical_daily = (
            historical_daily.reshape(-1, 1)
        )

    # -----------------------------------------------------------
    # RUN COMPLETE GENERATOR
    # -----------------------------------------------------------

    (
        synthetic_daily,
        synthetic_monthly
    ) = generate_synthetic_streamflow(
        historical_daily=historical_daily,
        number_of_realizations=
            number_of_realizations,
        number_of_years=
            number_of_synthetic_years,
        random_seed=random_seed
    )

    # -----------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------

    save_synthetic_csv(
        "synthetic_daily.csv",
        synthetic_daily
    )

    save_synthetic_csv(
        "synthetic_monthly.csv",
        synthetic_monthly
    )

    print("\nGeneration completed.")

    print(
        "Synthetic daily shape:",
        synthetic_daily.shape
    )

    print(
        "Synthetic monthly shape:",
        synthetic_monthly.shape
    )