"""
STEAD Dataset Loader Module.

This module provides a dataset loader class for the Stanford Earthquake Dataset (STEAD),
intended to work with the SeisT framework. It supports reading waveform data from an HDF5 file,
along with associated metadata from a CSV file, enabling training, validation, and test splits.

Classes:
    STEAD: Dataset class for reading and processing STEAD waveform and metadata.

Functions:
    stead: Factory function to register and return the STEAD dataset.
"""

import os
import h5py
import numpy as np
import pandas as pd
import threading
from typing import Tuple
from utils import logger
from .base import DatasetBase
from ._factory import register_dataset


class STEAD(DatasetBase):
    """
    Stanford Earthquake Dataset (STEAD) loader compatible with SeisT.

    This dataset loader reads waveform data from a HDF5 file and metadata from a CSV file.
    It prepares input in the format expected by the SeisT model, including waveform data
    and labels for tasks such as earthquake detection, P/S arrival time picking, magnitude
    regression, and azimuth estimation.

    Attributes:
        _name (str): Name of the dataset used for registry.
        _channels (List[str]): The seismic channels used (e.g., ['e', 'n', 'z']).
        _sampling_rate (int): The sampling rate of the data in Hz.
        _part_range (None): Placeholder for future use.
    """

    _name = "stead"
    _channels = ["e", "n", "z"]
    _sampling_rate = 100
    _part_range = None

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
        Initializes the STEAD dataset loader.

        Args:
            seed (int): Random seed for reproducibility.
            mode (str): One of ['train', 'val', 'test'] to select data split.
            data_dir (str): Path to the directory containing STEAD.hdf5 and STEAD.csv.
            shuffle (bool, optional): Whether to shuffle the metadata. Defaults to True.
            data_split (bool, optional): Whether to split into train/val/test. Defaults to True.
            train_size (float, optional): Proportion of data used for training. Defaults to 0.8.
            val_size (float, optional): Proportion of data used for validation. Defaults to 0.1.
            **kwargs: Additional arguments passed to the base class.
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
        self._h5f = None
        self._h5f_lock = threading.Lock()

    def __del__(self):
        """
        Ensures that the HDF5 file is properly closed when the object is destroyed.
        """
        if hasattr(self, "_h5f") and self._h5f:
            try:
                self._h5f.close()
            except Exception:
                pass

    def _get_h5f(self):
        """
        Lazily opens the HDF5 file in a thread-safe manner.

        Returns:
            h5py.File: An open HDF5 file handle.
        """
        with self._h5f_lock:
            if self._h5f is None:
                self._h5f = h5py.File(os.path.join(self._data_dir, "STEAD.hdf5"), "r")
        return self._h5f

    def _load_meta_data(self, filename="STEAD.csv") -> pd.DataFrame:
        """
        Loads and optionally shuffles and splits metadata from a CSV file.

        Args:
            filename (str): Name of the CSV file containing metadata. Defaults to 'STEAD.csv'.

        Returns:
            pd.DataFrame: A DataFrame containing the selected metadata entries.
        """
        meta_df = pd.read_csv(os.path.join(self._data_dir, filename), low_memory=False)

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, random_state=self._seed).reset_index(
                drop=True
            )

        if self._data_split:
            n = len(meta_df)
            train_end = int(n * self._train_size)
            val_end = train_end + int(n * self._val_size)
            splits = {
                "train": (0, train_end),
                "val": (train_end, val_end),
                "test": (val_end, n),
            }
            start, end = splits[self._mode]
            meta_df = meta_df.iloc[start:end]
            logger.info(f"STEAD Split: {self._mode}: {start}-{end}")

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """
        Loads waveform data and metadata for a specific event index.

        Args:
            idx (int): Index of the event in the metadata DataFrame.

        Returns:
            Tuple[dict, dict]: A tuple containing:
                - event (dict): Dictionary with waveform data and labels.
                - metadata (dict): Dictionary with full metadata for the event.
        """
        row = self._meta_data.iloc[idx]
        key = row["trace_name"]

        h5f = self._get_h5f()
        if "data" not in h5f or key not in h5f["data"]:
            raise KeyError(f"Key '{key}' not found in 'data/' group of HDF5.")

        data = h5f["data"][key][:].astype(np.float32)
        if data.shape[0] == 3:
            data = data.transpose(1, 0)

        label_map = {"earthquake_local": 1, "noise": 0}
        label = label_map.get(str(row.get("trace_category", "")).strip().lower(), -1)

        try:
            snr = np.array(eval(row["snr_db"]), dtype=np.float32)
        except Exception:
            snr = np.zeros(3, dtype=np.float32)

        event = {
            "data": data,
            "label": [label],
            "ppks": (
                [row["p_arrival_sample"]] if pd.notnull(row["p_arrival_sample"]) else []
            ),
            "spks": (
                [row["s_arrival_sample"]] if pd.notnull(row["s_arrival_sample"]) else []
            ),
            "emg": (
                [row["source_magnitude"]] if pd.notnull(row["source_magnitude"]) else []
            ),
            "pmp": [3],  # unknown polarity
            "clr": [0],  # no clarity field
            "baz": (
                [row["back_azimuth_deg"]] if pd.notnull(row["back_azimuth_deg"]) else []
            ),
            "snr": snr,
        }

        return event, row.to_dict()


@register_dataset
def stead(**kwargs):
    """
    Factory function to create and register the STEAD dataset.

    Args:
        **kwargs: Keyword arguments passed to the STEAD class.

    Returns:
        STEAD: An instance of the STEAD dataset loader.
    """
    return STEAD(**kwargs)
