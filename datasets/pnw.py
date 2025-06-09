"""
PNW Seismic Dataset Loader.

This module provides two dataset classes for handling the Pacific Northwest (PNW) seismic dataset:
PNW and PNW_light. It supports metadata parsing, waveform loading, and optional data splitting.

Reference:
    Ni, Y., Hutko, A., Skene, F., Denolle, M., Malone, S., Bodin, P., Hartog, R., & Wright, A. (2023).
    Curated Pacific Northwest AI-ready Seismic Dataset. Seismica, 2(1).
    https://doi.org/10.26443/seismica.v2i1.368
"""

from .base import DatasetBase
from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import h5py
from utils import logger
from ._factory import register_dataset


class PNW(DatasetBase):
    """PNW Dataset Loader.

    Loads seismic waveform data and metadata from the Pacific Northwest (PNW) dataset.
    Supports data shuffling, splitting, and preprocessing.

    Attributes:
        _name (str): Dataset name.
        _part_range (None): Placeholder, currently unused.
        _channels (List[str]): List of data channels used.
        _sampling_rate (int): Sampling rate of waveform data.
    """

    _name = "pnw"
    _part_range = None
    _channels = ["e", "n", "z"]
    _sampling_rate = 100

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the PNW dataset.

        Args:
            seed (int): Random seed for shuffling.
            mode (str): Mode of dataset: 'train', 'val', or 'test'.
            data_dir (str): Path to the dataset directory.
            shuffle (bool, optional): Whether to shuffle data. Defaults to True.
            data_split (bool, optional): Whether to split the dataset. Defaults to True.
            train_size (float, optional): Proportion of training data. Defaults to 0.8.
            val_size (float, optional): Proportion of validation data. Defaults to 0.1.
            **kwargs: Additional arguments.
        """
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )

    def _load_meta_data(self, filename="comcat_metadata.csv") -> pd.DataFrame:
        """
        Load metadata from a CSV file.

        Args:
            filename (str, optional): Name of the metadata CSV file. Defaults to "comcat_metadata.csv".

        Returns:
            pd.DataFrame: DataFrame containing metadata for events.
        """
        meta_df = pd.read_csv(
            os.path.join(self._data_dir, filename),
            low_memory=False,
        )

        for k in meta_df.columns:
            if meta_df[k].dtype in [np.dtype("float"), np.dtype("int")]:
                meta_df[k] = meta_df[k].fillna(0)
            elif meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].str.replace(" ", "")
                meta_df[k] = meta_df[k].fillna("")

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [
                irange["train"][1],
                irange["train"][1] + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0] : r[1], :]
            logger.info(f"Data Split: {self._mode}: {r[0]}-{r[1]}")

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """
        Load waveform and metadata for a single event.

        Args:
            idx (int): Index of the event.

        Raises:
            ValueError: If magnitude type is not 'ml'.

        Returns:
            Tuple[dict, dict]:
                - dict: Dictionary containing waveform data and labels.
                - dict: Dictionary of metadata fields for the event.
        """
        target_event = self._meta_data.iloc[idx]

        trace_name = target_event["trace_name"]
        bucket, array = trace_name.split("$")
        n, c, l = [int(i) for i in array.split(",:")]

        path = os.path.join(self._data_dir, f"comcat_waveforms.hdf5")
        with h5py.File(path, "r") as f:
            data = f.get(f"data/{bucket}")[n]
            data = np.array(data).astype(np.float32)
            data = np.nan_to_num(data)

        (
            ppk,
            spk,
            mag_type,
            evmag,
            motion,
            snr_str,
        ) = itemgetter(
            "trace_P_arrival_sample",
            "trace_S_arrival_sample",
            "preferred_source_magnitude_type",
            "preferred_source_magnitude",
            "trace_P_polarity",
            "trace_snr_db",
        )(target_event)

        motion = {"positive": 0, "negative": 1, "undecidable": 2, "": 3}[motion.lower()]

        assert mag_type.lower() == "ml"

        evmag = np.clip(evmag, 0, 8, dtype=np.float32)
        snrs = [s.strip() for s in snr_str.split("|")]
        snrs = [float(s) if s != "nan" else 0.0 for s in snrs]
        snr = np.array(snrs)

        event = {
            "data": data,
            "ppks": [ppk] if pd.notnull(ppk) else [],
            "spks": [spk] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [],
            "pmp": [motion] if pd.notnull(motion) else [],
            "clr": [0],  # For compatibility with other datasets
            "snr": snr,
        }

        return event, target_event.to_dict()


class PNW_light(PNW):
    """
    PNW_light Dataset Loader.

    Subclass of PNW that filters out events with undecidable P-polarity.

    Attributes:
        _name (str): Dataset name.
        _part_range (None): Placeholder, currently unused.
        _channels (List[str]): List of data channels used.
        _sampling_rate (int): Sampling rate of waveform data.
    """

    _name = "pnw_light"
    _part_range = None
    _channels = ["e", "n", "z"]
    _sampling_rate = 100

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the PNW_light dataset.

        Args:
            seed (int): Random seed for shuffling.
            mode (str): Mode of dataset: 'train', 'val', or 'test'.
            data_dir (str): Path to the dataset directory.
            shuffle (bool, optional): Whether to shuffle data. Defaults to True.
            data_split (bool, optional): Whether to split the dataset. Defaults to True.
            train_size (float, optional): Proportion of training data. Defaults to 0.8.
            val_size (float, optional): Proportion of validation data. Defaults to 0.1.
            **kwargs: Additional arguments.
        """
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )

    def _load_meta_data(self, filename="comcat_metadata_light.csv") -> pd.DataFrame:
        """
        Load filtered metadata (excluding undecidable P-polarity events).

        Args:
            filename (str, optional): Name of the filtered metadata CSV. Defaults to "comcat_metadata_light.csv".

        Returns:
            pd.DataFrame: Filtered metadata.
        """
        return super()._load_meta_data(filename=filename)

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """
        Load filtered event data (no undecidable polarity).

        Args:
            idx (int): Index of the event.

        Returns:
            Tuple[dict, dict]: Event data and metadata.
        """
        return super()._load_event_data(idx=idx)


@register_dataset
def pnw(**kwargs):
    """
    Register and return the standard PNW dataset.

    Returns:
        PNW: Instance of the PNW dataset.
    """
    dataset = PNW(**kwargs)
    return dataset


@register_dataset
def pnw_light(**kwargs):
    """
    Register and return the filtered PNW_light dataset.

    Returns:
        PNW_light: Instance of the PNW_light dataset.
    """
    dataset = PNW_light(**kwargs)
    return dataset
