"""
Module for the SOS waveform dataset.

This module defines the SOS dataset class which inherits from DatasetBase.
It provides methods to load metadata and waveform event data for the SOS
dataset, which contains seismic or similar waveform data sampled at 500 Hz
on a single channel "z".

Also includes a dataset factory registration function for easy instantiation.

Imports:
    DatasetBase: Base dataset class to inherit.
    os: For file path operations.
    pandas (pd): For metadata handling.
    numpy (np): For waveform data processing.
    logger, cal_snr: Utility functions for logging and SNR calculation.
    register_dataset: Decorator to register the dataset class.
"""

from .base import DatasetBase
import os 
from typing import Optional,Tuple
import pandas as pd
import numpy as np
from utils import logger,cal_snr
from ._factory import register_dataset



class SOS(DatasetBase):
    """
    SOS dataset class for waveform data handling.

    Inherits from DatasetBase to provide dataset loading and access functionality
    specific to the SOS dataset. The dataset contains waveform signals sampled at
    500 Hz from the "z" channel. Metadata and waveform events can be loaded
    separately. Supports optional dataset splitting and shuffling.

    Attributes:
        _name (str): Dataset name identifier, fixed as "sos".
        _part_range (NoneType): Not used, set to None.
        _channels (list[str]): List of channels, default ["z"].
        _sampling_rate (int): Sampling rate of waveform in Hz, default 500.
    """
    
    _name = "sos"
    _part_range = None
    _channels = ["z"]
    _sampling_rate = 500
    
    def __init__(
        self,
        seed:int,
        mode:str,
        data_dir:str,
        shuffle:bool=True,
        data_split:bool=False,
        train_size:float=0.8,
        val_size:float=0.1,
        **kwargs
        ):
        """
        Initialize the SOS dataset instance.

        Args:
            seed (int): Random seed for reproducible data shuffling.
            mode (str): Mode of the dataset, e.g. 'train', 'val', or 'test'.
            data_dir (str): Root directory containing the dataset files.
            shuffle (bool, optional): Whether to shuffle dataset entries. Defaults to True.
            data_split (bool, optional): Whether to split the dataset into train/val/test. Defaults to False.
            train_size (float, optional): Proportion of data to use for training if splitting. Defaults to 0.8.
            val_size (float, optional): Proportion of data to use for validation if splitting. Defaults to 0.1.
            **kwargs: Additional keyword arguments passed to DatasetBase.
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
            
    def _load_meta_data(self)->pd.DataFrame:
        """
        Load dataset metadata from CSV file.

        Loads a CSV file named '_all_label.csv' located under the directory
        path corresponding to the dataset mode. If data_split is enabled,
        issues a warning that this dataset has been pre-split and the argument
        will be ignored.

        Returns:
            pd.DataFrame: Metadata dataframe containing filename and label info.
        """
        if self._data_split:
            logger.warning(
                f"dataset 'sos' has been split into 'train','val' and 'test', argument 'data_split' will be ignored."
            )

        csv_path = os.path.join(self._data_dir, self._mode, "_all_label.csv")
        meta_df = pd.read_csv(
            csv_path, dtype={"fname": str, "itp": int, "its": int}
        )
        
        return meta_df
        
    def _load_event_data(self,idx:int) -> Tuple[dict,dict]:
        """
        Load waveform event data for the sample at the given index.

        Args:
            idx (int): Index of the target event in the metadata.

        Returns:
            Tuple[dict, dict]:
                - event (dict): Dictionary with keys:
                    "data" (np.ndarray): Waveform data array of shape (samples, channels).
                    "ppks" (list[int]): List containing primary P peak index if > 0, else empty.
                    "spks" (list[int]): List containing secondary S peak index if > 0, else empty.
                    "snr" (float): Signal-to-noise ratio calculated if primary peak present, else 0.0.
                - target_event (dict): Original metadata row converted to dictionary.
        """
        target_event = self._meta_data.iloc[idx]
        
        fname = target_event["fname"]
        ppk = target_event["itp"]
        spk = target_event["its"]

        fpath = os.path.join(self.data_dir, self.mode, fname)

        npz = np.load(fpath)

        data = npz["data"].astype(np.float32)
        
        data = np.stack(data, axis=1)

        event = {
            "data": data,
            "ppks": [ppk] if ppk > 0 else [],
            "spks": [spk] if spk > 0 else [],
            "snr": cal_snr(data=data,pat=ppk) if ppk > 0 else 0.
        }
        
        return event,target_event.to_dict()
    
@register_dataset
def sos(**kwargs):
    """
    Factory function to create an SOS dataset instance.

    This function instantiates and returns an SOS dataset object,
    passing all provided keyword arguments to the SOS constructor.

    Args:
        **kwargs: Arbitrary keyword arguments for SOS dataset initialization.

    Returns:
        SOS: Initialized SOS dataset instance.
    """
    dataset = SOS(**kwargs)
    return dataset