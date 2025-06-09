"""
This module defines the DatasetBase class, which serves as an abstract base class
for loading, managing, and accessing dataset samples and metadata.

The DatasetBase class provides a standardized interface for:
- Initializing datasets with reproducible splitting and shuffling options.
- Loading metadata about the dataset samples.
- Loading individual data samples by index.
- Managing dataset attributes such as name, sampling rate, and available channels.
- Supporting common dataset operations like length querying and string representation.

Subclasses should override the _load_meta_data and _load_event_data methods
to implement dataset-specific logic for metadata loading and sample retrieval.

Attributes:
    None (all class attributes are defined within DatasetBase and subclasses)

Usage example:
    class MyDataset(DatasetBase):
        def _load_meta_data(self, filename=None):
            # Implementation here
            pass

        def _load_event_data(self, idx):
            # Implementation here
            pass

    dataset = MyDataset(seed=42, mode="train", data_dir="data/")
    print(len(dataset))
    sample_input, sample_target = dataset[0]
"""

import pandas as pd
from typing import Optional, Tuple
import copy


class DatasetBase:
    """
    The base class for datasets.

    This class provides a template for dataset loading, metadata management, and basic access operations.
    It supports dataset splitting, shuffling, and metadata handling.

    Attributes:
        _name (str): The name of the dataset. Used to identify the dataset type or source.
        _part_range (Optional[tuple]): Specifies a range or subset of the dataset parts (e.g., indices or splits) that this dataset instance covers. Can be None if not applicable.
        _channels (list): A list of channel names or identifiers included in the dataset samples, e.g., ['RGB', 'Depth'].
        _sampling_rate (int): The sampling rate of the dataset data, usually in Hertz (Hz), indicating how frequently data samples are recorded or collected.
    """

    _name: str
    _part_range: Optional[tuple]
    _channels: list
    _sampling_rate: int

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
    ):
        """Initialize the base dataset.

        Args:
            seed (int): Random seed used for reproducibility.
            mode (str): Dataset mode, one of "train", "val", or "test".
            data_dir (str): Path to the dataset directory.
            shuffle (bool, optional): If True, the metadata will be shuffled. Defaults to True.
            data_split (bool, optional): If True, split data into train/val/test. Defaults to True.
            train_size (float, optional): Proportion of the dataset to use for training. Defaults to 0.8.
            val_size (float, optional): Proportion of the dataset to use for validation. Defaults to 0.1.

        Raises:
            AssertionError: If `mode` is not in ["train", "val", "test"].
            AssertionError: If the sum of `train_size` and `val_size` is >= 1.0.
        """
        self._seed = seed

        assert mode.lower() in ["train", "val", "test"]
        self._mode = mode.lower()

        self._data_dir = data_dir
        self._shuffle = shuffle
        self._data_split = data_split

        assert (
            train_size + val_size < 1.0
        ), f"train_size:{train_size}, val_size:{val_size}"
        self._train_size = train_size
        self._val_size = val_size

        self._meta_data = self._load_meta_data()

    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        """Load and return dataset metadata.

        Args:
            filename (str, optional): Optional filename for the metadata file.

        Returns:
            pd.DataFrame: A DataFrame containing metadata.

        Note:
            This method should be implemented by subclasses.
        """
        pass

    def _load_event_data(self, idx: int) -> dict:
        """Load event data for a given index.

        Args:
            idx (int): Index of the data sample to load.

        Returns:
            dict: A dictionary containing the sample data.

        Note:
            This method should be implemented by subclasses.
        """
        pass

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            str: Descriptive string with dataset attributes.
        """
        return (
            f"Dataset(name:{self._name}, part_range:{self._part_range}, channels:{self._channels}, "
            f"sampling_rate:{self._sampling_rate}, data_dir:{self._data_dir}, shuffle:{self._shuffle}, "
            f"data_split:{self._data_split}, train_size:{self._train_size}, val_size:{self._val_size})"
        )

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in metadata.
        """
        return len(self._meta_data)

    def __getitem__(self, idx: int) -> Tuple[dict, dict]:
        """Return a single data sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[dict, dict]: A tuple containing input and target data dictionaries.
        """
        return self._load_event_data(idx=idx)

    @classmethod
    def name(cls):
        """Return the dataset name.

        Returns:
            str: Name of the dataset.
        """
        return cls._name

    @classmethod
    def sampling_rate(cls):
        """Return the dataset sampling rate.

        Returns:
            int: Sampling rate in Hz.
        """
        return cls._sampling_rate

    @classmethod
    def channels(cls):
        """Return a copy of the list of channels.

        Returns:
            list: List of channel names.
        """
        return copy.deepcopy(cls._channels)
