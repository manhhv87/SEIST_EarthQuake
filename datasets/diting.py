"""
This module provides dataset classes for the DiTing seismic benchmark dataset.

It includes full and lightweight implementations designed for machine learning
tasks in seismology, supporting metadata loading, waveform access, and dataset
splits for training, validation, and testing.

Classes:
    DiTing: Loads metadata and waveform data from multiple CSV and HDF5 files,
        with options for shuffling and splitting the dataset.
    DiTing_light: A lightweight version of DiTing that uses a single CSV file
        for faster metadata loading.

Functions:
    diting(): Registers and returns an instance of the DiTing dataset.
    diting_light(): Registers and returns an instance of the DiTing_light dataset.

References:
    Zhao, M., Xiao, Z., Chen, S., & Fang, L. (2023).
    DiTing: A large-scale Chinese seismic benchmark dataset for artificial intelligence
    in seismology. Earthquake Science, 36(2), 84–94.
    https://doi.org/10.1016/j.eqs.2022.01.022
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


class DiTing(DatasetBase):
    """DiTing Dataset class.

    Provides access to the full DiTing seismic dataset, including waveform data
    and rich metadata. Supports train/val/test splitting and preprocessing.

    Attributes:
        _name (str): Dataset name identifier.
        _part_range (Tuple[int, int]): Range of data parts to load.
        _channels (List[str]): List of channel identifiers (e.g., ['z', 'n', 'e']).
        _sampling_rate (int): Sampling rate in Hz.
    """

    _name = "diting"
    _part_range = (0, 28)  # (inclusive,exclusive)
    _channels = ["z", "n", "e"]
    _sampling_rate = 50

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
        """Initializes the DiTing dataset.

        Args:
            seed (int): Random seed for reproducibility.
            mode (str): Dataset split mode, one of ['train', 'val', 'test'].
            data_dir (str): Path to the directory containing dataset files.
            shuffle (bool, optional): Whether to shuffle metadata. Defaults to True.
            data_split (bool, optional): Whether to split data into train/val/test. Defaults to True.
            train_size (float, optional): Proportion of training data. Defaults to 0.8.
            val_size (float, optional): Proportion of validation data. Defaults to 0.1.
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

    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        """Loads metadata for all parts in the specified part range.

        Args:
            filename (str, optional): CSV file to load. If None, loads default part files.

        Returns:
            pd.DataFrame: Loaded and optionally shuffled/split metadata.
        """
        start, end = self._part_range
        meta_df = pd.concat(
            [
                pd.read_csv(
                    os.path.join(self._data_dir, f"DiTing330km_part_{i}.csv"),
                    dtype={
                        "part": np.int64,
                        "key": str,
                        "ev_id": np.int64,
                        "evmag": str,
                        "mag_type": str,
                        "p_pick": np.int64,
                        "p_clarity": str,
                        "p_motion": str,
                        "s_pick": np.int64,
                        "net": str,
                        "sta_id": np.int64,
                        "dis": np.float32,
                        "st_mag": str,
                        "baz": str,
                        "Z_P_amplitude_snr": np.float32,
                        "Z_P_power_snr": np.float32,
                        "Z_S_amplitude_snr": np.float32,
                        "Z_S_power_snr": np.float32,
                        "N_P_amplitude_snr": np.float32,
                        "N_P_power_snr": np.float32,
                        "N_S_amplitude_snr": np.float32,
                        "N_S_power_snr": np.float32,
                        "E_P_amplitude_snr": np.float32,
                        "E_P_power_snr": np.float32,
                        "E_S_amplitude_snr": np.float32,
                        "E_S_power_snr": np.float32,
                        "P_residual": str,
                        "S_residual": str,
                    },
                    low_memory=False,
                    index_col=0,
                )
                for i in range(start, end)
            ]
        )

        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].str.replace(" ", "")

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
        """Loads seismic event data and metadata.

        Args:
            idx (int): Index of the event in the metadata dataframe.

        Raises:
            ValueError: If the magnitude type is unknown.

        Returns:
            Tuple[dict, dict]: A tuple containing:
                - dict with waveform data and preprocessed features (e.g. magnitude, SNR).
                - dict with raw metadata for the event.
        """

        target_event = self._meta_data.iloc[idx]
        part = target_event["part"]
        key = target_event["key"]
        key_correct = key.split(".")
        key = key_correct[0].rjust(6, "0") + "." + key_correct[1].ljust(4, "0")

        path = os.path.join(self._data_dir, f"DiTing330km_part_{part}.hdf5")
        with h5py.File(path, "r") as f:
            dataset = f.get("earthquake/" + str(key))
            data = np.array(dataset).astype(np.float32).T

        (
            ppk,
            spk,
            mag_type,
            evmag,
            stmag,
            motion,
            clarity,
            baz,
            dis,
            zpp_snr,
            nsp_snr,
            esp_snr,
        ) = itemgetter(
            "p_pick",
            "s_pick",
            "mag_type",
            "evmag",
            "st_mag",
            "p_motion",
            "p_clarity",
            "baz",
            "dis",
            "Z_P_power_snr",
            "N_S_power_snr",
            "E_S_power_snr",
        )(
            target_event
        )

        if pd.notnull(motion) and motion.lower() not in ["", "n"]:
            motion = {"u": 0, "c": 0, "r": 1, "d": 1}[motion.lower()]

        if pd.notnull(clarity):
            clarity = 0 if clarity.lower() == "i" else 1

        if pd.notnull(baz):
            baz = baz % 360

        mag_type_lower = mag_type.lower()
        # To ml
        if mag_type_lower == "ms":
            evmag = (evmag + 1.08) / 1.13
            stmag = (stmag + 1.08) / 1.13
        elif mag_type_lower == "mb":
            evmag = (1.17 * evmag + 0.67) / 1.13
            stmag = (1.17 * stmag + 0.67) / 1.13
        elif mag_type_lower == "ml":
            pass
        else:
            raise ValueError(f"Unknown 'mag_type' : '{mag_type}'")

        evmag = np.clip(evmag, 0, 8, dtype=np.float32)
        stmag = np.clip(stmag, 0, 8, dtype=np.float32)

        snr = np.array([zpp_snr, nsp_snr, esp_snr])

        event = {
            "data": data,
            "ppks": [ppk] if pd.notnull(ppk) else [],
            "spks": [spk] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [],
            "smg": [stmag] if pd.notnull(stmag) else [],
            "pmp": [motion] if pd.notnull(motion) else [],
            "clr": [clarity] if pd.notnull(clarity) else [],
            "baz": [baz] if pd.notnull(baz) else [],
            "dis": [dis] if pd.notnull(dis) else [],
            "snr": snr,
        }

        return event, target_event.to_dict()


class DiTing_light(DiTing):
    """Light version of the DiTing Dataset.

    A lightweight version of the DiTing dataset that loads metadata from a
    single file instead of from multiple parts.
    """

    _name = "diting_light"
    _part_range = None
    _channels = ["z", "n", "e"]
    _sampling_rate = 50

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
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )

    def _load_meta_data(self, filename=f"DiTing330km_light.csv") -> pd.DataFrame:
        """Loads metadata from a single CSV file.

        Args:
            filename (str): Name of the CSV file containing metadata.

        Returns:
            pd.DataFrame: Loaded and optionally shuffled/split metadata.
        """

        meta_df = pd.read_csv(
            os.path.join(self._data_dir, filename),
            dtype={
                "part": np.int64,
                "key": str,
                "ev_id": np.int64,
                "evmag": np.float32,
                "mag_type": str,
                "p_pick": np.int64,
                "p_clarity": str,
                "p_motion": str,
                "s_pick": np.int64,
                "net": str,
                "sta_id": np.int64,
                "dis": np.float32,
                "st_mag": np.float32,
                "baz": np.float32,
                "Z_P_amplitude_snr": np.float32,
                "Z_P_power_snr": np.float32,
                "Z_S_amplitude_snr": np.float32,
                "Z_S_power_snr": np.float32,
                "N_P_amplitude_snr": np.float32,
                "N_P_power_snr": np.float32,
                "N_S_amplitude_snr": np.float32,
                "N_S_power_snr": np.float32,
                "E_P_amplitude_snr": np.float32,
                "E_P_power_snr": np.float32,
                "E_S_amplitude_snr": np.float32,
                "E_S_power_snr": np.float32,
                "P_residual": np.float32,
                "S_residual": np.float32,
            },
            low_memory=False,
            index_col=0,
        )

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
        """Loads seismic event data and metadata.

        Args:
            idx (int): Index of the event in the metadata dataframe.

        Returns:
            Tuple[dict, dict]: A tuple containing:
                - dict with waveform data and preprocessed features.
                - dict with raw metadata for the event.
        """
        return super()._load_event_data(idx=idx)


@register_dataset
def diting(**kwargs):
    """Dataset registration for DiTing.

    Args:
        **kwargs: Arguments to initialize the DiTing dataset.

    Returns:
        DiTing: An instance of the DiTing dataset.
    """
    dataset = DiTing(**kwargs)
    return dataset


@register_dataset
def diting_light(**kwargs):
    """Dataset registration for DiTing_light.

    Args:
        **kwargs: Arguments to initialize the DiTing_light dataset.

    Returns:
        DiTing_light: An instance of the DiTing_light dataset.
    """
    dataset = DiTing_light(**kwargs)
    return dataset
