from .base import DatasetBase
from ._factory import register_dataset
import os
import pandas as pd
import numpy as np
import h5py
from typing import Tuple
from utils import logger


class STEAD(DatasetBase):
    """Stanford Earthquake Dataset (STEAD) loader compatible with SeisT.

    This dataset loader reads waveform data from a HDF5 file and metadata from a CSV file.
    It prepares input in the format expected by the SeisT model, including waveform data
    and labels for tasks such as earthquake detection, P/S arrival time picking, magnitude
    regression, and azimuth estimation.
    """

    _name = "stead"
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
        """Initializes the STEAD dataset instance.

        Args:
            seed (int): Random seed for reproducibility.
            mode (str): One of ['train', 'val', 'test'].
            data_dir (str): Path to the dataset directory.
            shuffle (bool): Whether to shuffle metadata.
            data_split (bool): Whether to perform train/val/test split.
            train_size (float): Proportion of training data.
            val_size (float): Proportion of validation data.
            **kwargs: Reserved.
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

    def _load_meta_data(self, filename="STEAD.csv") -> pd.DataFrame:
        """Loads and optionally splits metadata from a CSV file.

        Args:
            filename (str): Name of the CSV file. Default is "STEAD.csv".

        Returns:
            pd.DataFrame: Subset metadata matching current mode.
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
        """Loads one waveform and its metadata for SeisT model input.

        Args:
            idx (int): Index in metadata DataFrame.

        Returns:
            Tuple[dict, dict]: Event dictionary and raw metadata.
        """
        row = self._meta_data.iloc[idx]
        key = row["trace_name"]

        hdf5_path = os.path.join(self._data_dir, "STEAD.hdf5")
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise KeyError("Group 'data' not found in the HDF5 file.")

            if key not in f["data"]:
                raise KeyError(f"Key '{key}' not found in 'data/' group of HDF5.")

            data = f["data"][key][:].astype(np.float32)

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
            "pmp": [3],  # Unknown polarity
            "clr": [0],  # No clarity field
            "baz": (
                [row["back_azimuth_deg"]] if pd.notnull(row["back_azimuth_deg"]) else []
            ),
            "snr": snr,
        }

        return event, row.to_dict()


@register_dataset
def stead(**kwargs):
    """Factory function to register the STEAD dataset.

    Args:
        **kwargs: Keyword arguments for STEAD constructor.

    Returns:
        STEAD: Dataset instance.
    """
    return STEAD(**kwargs)
