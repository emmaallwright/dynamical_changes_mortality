"""Plot 1 of dynamical changes and mortality.

This script loads mortality and counterfactual data for Europe and selected
countries, then generates a multi-panel figure showing summer values for
2018 and 2003.

The workflow is structured into small helper functions for dataset loading,
region selection, and plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

BASE_DIR = Path("path")
SHAPE_DIR = Path("path")
OUTPUT_PATH = BASE_DIR / "figures" / "hrm_case_study_all_fr_es_it_v2.png"
DATA_DIR = BASE_DIR / "mortality_data"

# Mapping from dataset keys to filenames
FILENAME_MAPPING = {
    "cf_pos": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_positive_Z500_AF_TOT.nc",
    "cf_neg": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_negative_Z500_AF_TOT.nc",
    "cf_both": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_both_Z500_AF_TOT.nc",
    "f_pos": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_positive_Z500_AF_TOT.nc",
    "f_neg": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_negative_Z500_AF_TOT.nc",
    "f_both": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_both_Z500_AF_TOT.nc",
    "cf_pos_cou": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_positive_Z500_AF_REG.nc",
    "cf_neg_cou": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_negative_Z500_AF_REG.nc",
    "f_pos_cou": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_positive_Z500_AF_REG.nc",
    "f_neg_cou": "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.factual_negative_Z500_AF_REG.nc",
}

EXCEL_FILE = (
    DATA_DIR
    / "TABLE_WEEKS_AF_totalheat_allsex_allage_iSENSITIVITY.0_sANALOGUE.true_factual_COU.xlsx"
)
CSV_FILE = (
    DATA_DIR
    / "figure1_allsex_allage_iSENSITIVITY.0_sANALOGUE.counterfactual_both_Z500_METATABLE.csv"
)

# Regional slices used for France, Spain, Italy
REGION_SLICES = {
    "FR": ("FR101", "FRM01"),
    "ES": ("ES111", "ES709"),
    "IT": ("ITC11", "ITI45"),
}

SUBTITLES = [
    "a) 2018 Europe",
    "b) 2003 Europe",
    "c) 2018 France",
    "d) 2003 France",
    "e) 2018 Spain",
    "f) 2003 Spain",
    "g) 2018 Italy",
    "h) 2003 Italy",
]

SERIES_YEARS = ["2018", "2003", "2018", "2003", "2018", "2003"]
LINE_COLORS = ["black", "black", "red", "red", "blue", "blue"]
LINE_LABELS = [
    "Factual",
    "Factual",
    "1950 counterfactual",
    "1950 counterfactual",
    "Trended counterfactual",
    "Trended counterfactual",
]


def open_datasets() -> dict[str, xr.Dataset]:
    """Load all required mortality NetCDF datasets.

    Returns:
        A dictionary mapping dataset keys to xarray datasets.
    """
    return {
        key: xr.open_dataset(DATA_DIR / filename)
        for key, filename in FILENAME_MAPPING.items()
    }


def load_shapes() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load geographic shape files used for region selection.

    Returns:
        A tuple containing the Europe NUTS shape and the country boundary shape.
    """
    europe_shape = gpd.read_file(SHAPE_DIR / "europe_allnuts_gisco_2003_2021_nuts3_shap.geojson")
    country_shape = gpd.read_file(SHAPE_DIR / "CNTR_RG_10M_2020_3035.geojson")
    return europe_shape, country_shape


def load_time_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the observed country and Europe time series from Excel.

    Returns:
        Four NumPy arrays for Europe, France, Italy, and Spain observed values.
    """
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    europe = df.iloc[0, 2:].values
    france = df.iloc[14, 2:].values
    italy = df.iloc[19, 2:].values
    spain = df.iloc[12, 2:].values
    return europe, france, spain, italy


def assign_region_dim(data: xr.Dataset, region_ids: np.ndarray) -> xr.Dataset:
    """Attach explicit region identifiers to the dataset dim1 coordinate.

    Args:
        data: The input xarray dataset that lacks region IDs on dim1.
        region_ids: An array of region identifiers from the metadata CSV.

    Returns:
        The same dataset with dim1 assigned to the provided region IDs.
    """
    return data.assign(dim1=region_ids)


def region_mean(data: xr.Dataset, region_slice: tuple[str, str]) -> xr.DataArray:
    """Compute the region-averaged data array for a country slice.

    Args:
        data: The regional dataset with dim1 representing region IDs.
        region_slice: A tuple of start/end region codes to select the country.

    Returns:
        The mean DataArray over the selected dim1 slice.
    """
    return (
        data.isel(dim2=slice(0, -1), dim3=2, dim4=0)
        .sel(dim1=slice(*region_slice))
        .mean("dim1")
    )


def region_series(base_series: np.ndarray, data: xr.Dataset) -> xr.DataArray:
    """Create a dim2-aligned DataArray from a base time series.

    Args:
        base_series: A NumPy array containing the observed time series values.
        data: The dataset or DataArray whose dim2 coordinate will be used for alignment.

    Returns:
        An xarray DataArray indexed by dim2, with the same length as the data selection.
    """
    if isinstance(data, xr.Dataset):
        series = data["var1_1"]
    else:
        series = data

    if "dim3" in series.dims and "dim4" in series.dims:
        series = series.isel(dim2=slice(0, -1), dim3=2, dim4=0)
    else:
        series = series.isel(dim2=slice(0, -1))

    base_array = np.asarray(base_series)
    target_length = series.sizes["dim2"]
    if base_array.shape[0] != target_length:
        if base_array.shape[0] > target_length:
            base_array = base_array[:target_length]
        else:
            raise ValueError(
                f"Base series length {base_array.shape[0]} does not match dim2 length {target_length}."
            )

    return xr.DataArray(
        base_array,
        coords={"dim2": series["dim2"].values},
        dims=["dim2"],
    )


def build_plot_data(
    europe_ts: np.ndarray,
    france_ts: np.ndarray,
    spain_ts: np.ndarray,
    italy_ts: np.ndarray,
    datasets: dict[str, xr.Dataset],
    region_ids: np.ndarray,
) -> list[list[xr.DataArray]]:
    """Build all series arrays required for plotting.

    Args:
        europe_ts: Observed Europe time series array.
        france_ts: Observed France time series array.
        spain_ts: Observed Spain time series array.
        italy_ts: Observed Italy time series array.
        datasets: Loaded mortality datasets keyed by filename alias.
        region_ids: The region ID values to assign to regional datasets.

    Returns:
        A list of four row lists, each containing six DataArrays to plot.
    """
    cf_pos = datasets["cf_pos"]
    f_pos = datasets["f_pos"]
    cf_pos_cou = assign_region_dim(datasets["cf_pos_cou"], region_ids)
    f_pos_cou = assign_region_dim(datasets["f_pos_cou"], region_ids)

    europe_data = [
        region_series(europe_ts, cf_pos),
        region_series(europe_ts, cf_pos),
        cf_pos.isel(dim1=0, dim2=slice(0, -1), dim3=2, dim4=0),
        cf_pos.isel(dim1=0, dim2=slice(0, -1), dim3=2, dim4=0),
        f_pos.isel(dim1=0, dim2=slice(0, -1), dim3=2, dim4=0),
        f_pos.isel(dim1=0, dim2=slice(0, -1), dim3=2, dim4=0),
    ]

    france_data = [
        region_series(france_ts, region_mean(cf_pos_cou, REGION_SLICES["FR"])),
        region_series(france_ts, region_mean(cf_pos_cou, REGION_SLICES["FR"])),
        region_mean(cf_pos_cou, REGION_SLICES["FR"]),
        region_mean(cf_pos_cou, REGION_SLICES["FR"]),
        region_mean(f_pos_cou, REGION_SLICES["FR"]),
        region_mean(f_pos_cou, REGION_SLICES["FR"]),
    ]

    spain_data = [
        region_series(spain_ts, region_mean(cf_pos_cou, REGION_SLICES["ES"])),
        region_series(spain_ts, region_mean(cf_pos_cou, REGION_SLICES["ES"])),
        region_mean(cf_pos_cou, REGION_SLICES["ES"]),
        region_mean(cf_pos_cou, REGION_SLICES["ES"]),
        region_mean(f_pos_cou, REGION_SLICES["ES"]),
        region_mean(f_pos_cou, REGION_SLICES["ES"]),
    ]

    italy_data = [
        region_series(italy_ts, region_mean(cf_pos_cou, REGION_SLICES["IT"])),
        region_series(italy_ts, region_mean(cf_pos_cou, REGION_SLICES["IT"])),
        region_mean(cf_pos_cou, REGION_SLICES["IT"]),
        region_mean(cf_pos_cou, REGION_SLICES["IT"]),
        region_mean(f_pos_cou, REGION_SLICES["IT"]),
        region_mean(f_pos_cou, REGION_SLICES["IT"]),
    ]

    return [europe_data, france_data, spain_data, italy_data]
def create_week_index() -> pd.DatetimeIndex:
    """Create the weekly datetime index used for plotting.

    Returns:
        A pandas DatetimeIndex with weekly frequency from 1950-01-01 to 2021-12-02.
    """
    return pd.date_range(start="1950-01-01", end="2021-12-02", freq="W")


def plot_series(plot_data: list[list[xr.DataArray]]) -> None:
    """Plot the six series for Europe and selected countries.

    Args:
        plot_data: Nested list of DataArrays for each row and subplot.
    """
    week_index = create_week_index()
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(35, 55))

    for row_index, region_data in enumerate(plot_data):
        for item_index, data_array in enumerate(region_data):
            ax = axes[row_index, item_index % 2]
            data_array = data_array.copy()
            data_len = len(data_array["dim2"])
            if data_len != len(week_index):
                data_array["dim2"] = week_index[:data_len]
            else:
                data_array["dim2"] = week_index

            if isinstance(data_array, xr.Dataset):
                series = data_array["var1_1"]
            else:
                series = data_array

            selected = series.sel(
                dim2=slice(f"{SERIES_YEARS[item_index]}-06", f"{SERIES_YEARS[item_index]}-08")
            )

            ax.plot(
                selected.dim2,
                selected.values,
                color=LINE_COLORS[item_index],
                alpha=0.9,
                linewidth=5,
                label=LINE_LABELS[item_index],
            )

            if item_index < 2:
                baseline = series.sel(dim2=series.dim2.dt.month.isin([6, 7, 8]))
                ax.axhline(
                    baseline.mean(),
                    alpha=0.9,
                    linewidth=5,
                    color="black",
                    linestyle="--",
                )
                ax.set_title(SUBTITLES[row_index * 2 + item_index], fontsize=34)
                ax.set_xlabel("time (date)", fontsize=28)
                ax.set_ylabel("AF (%)", fontsize=28)
                summer_ticks = selected.dim2[selected.dim2.dt.month.isin([6, 7, 8])][::4]
                ax.set_xticks(summer_ticks.values)
                ax.set_xticklabels(summer_ticks.dt.strftime("%m-%d").values, fontsize=18)
                ax.tick_params(size=28, labelsize=28)
                ax.set_ylim(0, 22)

    axes[3, 0].legend(
        loc="upper center",
        bbox_to_anchor=(1.1, -0.15),
        fancybox=True,
        ncol=3,
        fontsize=28,
    )

    fig.subplots_adjust(
        left=0.05,
        bottom=0.22,
        right=0.9,
        top=0.9,
        wspace=0.14,
        hspace=0.22,
    )
    fig.savefig(OUTPUT_PATH)


def main() -> None:
    """Main entry point that loads data and renders the figure.

    The script loads datasets, prepares plot data, and saves the result to
    the configured output path.
    """
    datasets = open_datasets()
    load_shapes()
    europe_ts, france_ts, spain_ts, italy_ts = load_time_series()
    regions = pd.read_csv(CSV_FILE)["location"].values
    plot_data = build_plot_data(
        europe_ts,
        france_ts,
        spain_ts,
        italy_ts,
        datasets,
        regions,
    )
    plot_series(plot_data)


if __name__ == "__main__":
    main()
