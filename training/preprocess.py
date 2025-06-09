import argparse
import copy
from utils import logger
from config import Config
from operator import itemgetter
from typing import Any, List, Tuple, Union, Union
import numpy as np
import json
from datasets import build_dataset
from torch.utils.data import Dataset


__all__ = ["Preprocessor", "SeismicDataset"]


def _pad_phases(
    ppks: list, spks: list, padding_idx: int, num_samples: int
) -> Tuple[list, list]:
    """
    Pad P-phase and S-phase pick lists so that both have the same length.

    This function ensures the two input lists, representing detected P-phase (ppks)
    and S-phase (spks) picks, have equal lengths by prepending/appending
    padding values. It uses the relative ordering of picks to determine the amount
    of padding needed.

    Args:
        ppks (list): List of P-phase pick sample indices (integers).
        spks (list): List of S-phase pick sample indices (integers).
        padding_idx (int): Integer used as padding index (absolute value used).
        num_samples (int): Total number of samples in the waveform, used to
            generate padding values for S-phase picks.

    Returns:
        Tuple[list, list]: Tuple containing two lists of equal length:
            - P-phase picks padded with negative padding indices at the start.
            - S-phase picks padded with large indices (beyond num_samples)
              at the end.

    Raises:
        AssertionError: If after padding the two lists do not have equal length.

    Example:
        >>> _pad_phases([10, 50], [20], padding_idx=1, num_samples=100)
        ([-1, 10, 50], [20, 101, 101])

    Notes:
        - Padding indices for P-phase picks are negative values.
        - Padding indices for S-phase picks are values greater than `num_samples`.
        - This padding facilitates batch processing by keeping phase pick lists
          aligned in length.
    """
    padding_idx = abs(padding_idx)
    ppks, spks = sorted(ppks), sorted(spks)
    ppks_, spks_ = ppks.copy(), spks.copy()
    ppk_arr, spk_arr = np.array(ppks), np.array(sorted(spks))
    idx = 0
    while idx < min(len(ppks), len(spks)) and all(
        ppk_arr[: idx + 1] < spk_arr[-idx - 1 :]
    ):
        idx += 1
    ppks = len(spk_arr[: len(spk_arr) - idx]) * [-padding_idx] + ppks
    spks = spks + len(ppk_arr[idx:]) * [num_samples + padding_idx]

    assert len(ppks) == len(spks), f"Error:{ppks_} -> {ppks},{spks_} -> {spks}"
    return ppks, spks


def _pad_array(s: list, length: int, padding_value: Union[int, float]) -> np.ndarray:
    """
    Pad a list or array to a specified length using a given padding value.

    This function pads the input list `s` with the specified `padding_value` at the
    end, so that the returned array has the exact length `length`. If the input list
    is already longer than `length`, an exception is raised.

    Args:
        s (list): Input list to pad.
        length (int): Target length of the padded array.
        padding_value (int or float): Value used to pad the array.

    Returns:
        np.ndarray: Padded array of length `length`.

    Raises:
        Exception: If `length` is smaller than the length of input list `s`.
    """
    padding_size = int(length - len(s))
    if padding_size >= 0:
        padded = np.pad(
            s, (0, padding_size), mode="constant", constant_values=padding_value
        )
        return padded
    else:
        raise Exception(f"`length < len(s)` . Array:{len(s)},Target:{length}")


class DataPreprocessor:
    """
    Data preprocessor for input data augmentation and label generation.

    This class handles preprocessing of input data including normalization, amplitude adjustment,
    and data augmentation techniques such as adding noise, dropping channels, shifting events, etc.
    It also generates labels based on the processed data.

    Some data augmentation methods (e.g. `_normalize`, `_adjust_amplitude`, `_scale_amplitude`, `_pre_emphasis`)
    are adapted from:
    https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/EqT_uilts.py

    Args:
        data_channels (int): Number of data channels in the input.
        sampling_rate (int): Sampling rate of the input data (samples per second).
        in_samples (int): Number of input samples.
        min_snr (float): Minimum signal-to-noise ratio required for augmentation.
        p_position_ratio (float): Ratio used for P-phase position augmentation (between 0 and 1).
        coda_ratio (float): Ratio to define the length of coda relative to event length.
        norm_mode (str): Normalization mode to be applied to the data.
        add_event_rate (float): Probability of adding an event during augmentation.
        add_noise_rate (float): Probability of adding noise during augmentation.
        add_gap_rate (float): Probability of adding gaps in the data.
        drop_channel_rate (float): Probability of dropping a data channel.
        scale_amplitude_rate (float): Probability of scaling the amplitude.
        pre_emphasis_rate (float): Probability of applying pre-emphasis filter.
        pre_emphasis_ratio (float): Ratio parameter for pre-emphasis.
        max_event_num (float): Maximum number of events to be generated.
        generate_noise_rate (float): Probability of generating noise-only samples.
        shift_event_rate (float): Probability of shifting events in time.
        mask_percent (float): Percentage of the input data to mask.
        noise_percent (float): Percentage of noise to add in augmentation.
        min_event_gap_sec (float): Minimum gap between events in seconds.
        soft_label_shape (str): Shape type of the soft labels.
        soft_label_width (int): Width parameter for soft labels.
        dtype (np.dtype, optional): Data type for the processed data. Defaults to np.float32.

    Raises:
        Warning: Logs warnings and disables certain augmentations when `p_position_ratio` is between 0 and 1,
                 as it conflicts with adding events, shifting events, and generating noise.
    """

    def __init__(
        self,
        data_channels: int,
        sampling_rate: int,
        in_samples: int,
        min_snr: float,
        p_position_ratio: float,
        coda_ratio: float,
        norm_mode: str,
        add_event_rate: float,
        add_noise_rate: float,
        add_gap_rate: float,
        drop_channel_rate: float,
        scale_amplitude_rate: float,
        pre_emphasis_rate: float,
        pre_emphasis_ratio: float,
        max_event_num: float,
        generate_noise_rate: float,
        shift_event_rate: float,
        mask_percent: float,
        noise_percent: float,
        min_event_gap_sec: float,
        soft_label_shape: str,
        soft_label_width: int,
        dtype=np.float32,
    ):
        self.sampling_rate = sampling_rate

        self.data_channels = data_channels

        self.in_samples = in_samples
        self.coda_ratio = coda_ratio
        self.norm_mode = norm_mode
        self.min_snr = min_snr
        self.p_position_ratio = p_position_ratio

        self.add_event_rate = add_event_rate
        self.add_noise_rate = add_noise_rate
        self.add_gap_rate = add_gap_rate
        self.drop_channel_rate = drop_channel_rate
        self.scale_amplitude_rate = scale_amplitude_rate
        self.pre_emphasis_rate = pre_emphasis_rate
        self.pre_emphasis_ratio = pre_emphasis_ratio
        self._max_event_num = max_event_num
        self.generate_noise_rate = generate_noise_rate
        self.shift_event_rate = shift_event_rate
        self.mask_percent = mask_percent
        self.noise_percent = noise_percent
        self.min_event_gap = int(min_event_gap_sec * self.sampling_rate)

        if 0 <= self.p_position_ratio <= 1:
            if self.add_event_rate > 0:
                self.add_event_rate = 0.0
                logger.warning(
                    f"`p_position_ratio` is {p_position_ratio}, `add_event_rate` -> `0.0`"
                )

            if self.shift_event_rate > 0:
                self.shift_event_rate = 0.0
                logger.warning(
                    f"`p_position_ratio` is {p_position_ratio}, `shift_event_rate` -> `0.0`"
                )

            if self.generate_noise_rate > 0:
                self.generate_noise_rate = 0.0
                logger.warning(
                    f"`p_position_ratio` is {p_position_ratio}, `generate_noise_rate` -> `0.0`"
                )

        self.soft_label_shape = soft_label_shape
        self.soft_label_width = soft_label_width
        self.dtype = dtype

    def _clear_dict_except(self, d: dict, *args) -> None:
        """
        Clear the contents of a dictionary except for specified keys.

        This method clears or resets the values in the dictionary `d` for all keys
        except those specified in `args`. The clearing behavior depends on the
        value type:
        - For `list` or `dict`, it clears the contents.
        - For `np.ndarray`, it resets to an empty numpy array.
        - For `int` or `float`, it resets to 0.
        - For `str`, it resets to an empty string.
        - For other types, raises a TypeError.

        Args:
            d (dict): The dictionary to clear values from.
            *args (str): Keys to exclude from clearing. Must be strings.

        Raises:
            AssertionError: If any of the keys in `args` is not a string.
            TypeError: If a dictionary value has an unsupported type.
        """
        if len(args) > 0:
            for arg in args:
                assert isinstance(
                    arg, str
                ), f"Input arguments must be str, got `{arg}`({type(arg)})"
        for k in set(d) - set(args):
            if isinstance(d[k], (list, dict)):
                d[k].clear()
            elif isinstance(d[k], np.ndarray):
                d[k] = np.array([])
            elif isinstance(d[k], (int, float)):
                d[k] = 0
            elif isinstance(d[k], str):
                d[k] = ""
            else:
                raise TypeError(f"Got `{d[k]}`({type(d[k])})")

    def _is_noise(
        self, data: np.ndarray, ppks: List[int], spks: List[int], snr: np.ndarray
    ) -> bool:
        """
        Determine whether the given data segment should be classified as noise.

        The function checks several conditions to decide if the input data
        represents noise rather than a valid event:
        - The number of P-phase picks (`ppks`) and S-phase picks (`spks`) must be equal and non-zero.
        - The indices in `ppks` and `spks` must be within valid data bounds.
        - The signal-to-noise ratio (`snr`) must meet a minimum threshold.
        - Each P-phase pick must occur before its corresponding S-phase pick.

        Args:
            data (np.ndarray): Input data array.
            ppks (List[int]): List of P-phase pick indices.
            spks (List[int]): List of S-phase pick indices.
            snr (np.ndarray): Signal-to-noise ratio array.

        Returns:
            bool: True if data is considered noise, False otherwise.
        """
        is_noise = (
            (len(ppks) != len(spks))
            or len(ppks) < 1
            or len(spks) < 1
            or min(ppks + spks) < 0
            or max(ppks + spks) >= data.shape[-1]
            or all(snr < self.min_snr)
        )
        for i in range(len(ppks)):
            is_noise |= ppks[i] >= spks[i]
        return is_noise

    def _cut_window(
        self, data: np.ndarray, ppks: list, spks: list, window_size: int
    ) -> Tuple[np.ndarray, list, list]:
        """
        Slice or pad the input data to a fixed window size and adjust event picks accordingly.

        Depending on the `p_position_ratio` parameter, the function either centers the window
        around the first P-phase pick or randomly slices the data. It adjusts the P- and S-phase
        pick indices to fit within the sliced window and pads the data with zeros if it is shorter
        than the desired window size.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            ppks (list): List of P-phase pick indices.
            spks (list): List of S-phase pick indices.
            window_size (int): Desired output window size (number of samples).

        Returns:
            Tuple[np.ndarray, list, list]: Tuple containing
                - The sliced (or padded) data array of shape (channels, window_size).
                - Adjusted list of P-phase pick indices within the new window.
                - Adjusted list of S-phase pick indices within the new window.
        """
        input_len = data.shape[-1]

        if 0 <= self.p_position_ratio <= 1:
            new_data = np.zeros((data.shape[0], window_size), dtype=np.float32)
            tgt_l, tgt_r = 0, window_size

            p_idx = ppks[0]
            c_l = p_idx - int(window_size * self.p_position_ratio)
            c_r = c_l + window_size
            offset = -c_l

            if c_l < 0:
                tgt_l += abs(c_l)
                offset += c_l
                c_l = 0

            if c_r > data.shape[-1]:
                tgt_r -= c_r - data.shape[-1]
                c_r = data.shape[-1]

            new_data[:, tgt_l:tgt_r] = data[:, c_l:c_r]
            offset += tgt_l
            data = new_data

            ppks = [t + offset for t in ppks if 0 <= t + offset < window_size]
            spks = [t + offset for t in spks if 0 <= t + offset < window_size]

        else:
            if input_len > window_size:
                c_l = np.random.randint(
                    0,
                    max(min(ppks + [input_len - window_size]) - self.min_event_gap, 1),
                )
                c_r = c_l + window_size

                data = data[:, c_l:c_r]
                ppks = [t - c_l for t in ppks if c_l <= t < c_r]
                spks = [t - c_l for t in spks if c_l <= t < c_r]

            elif input_len < window_size:
                data = np.concatenate(
                    [data, np.zeros((data.shape[0], window_size - input_len))], axis=1
                )

        return data, ppks, spks

    def _normalize(self, data, mode):
        """
        Normalize the waveform of each sample in-place.

        The normalization subtracts the mean of each channel and then applies either
        max normalization or standard deviation normalization based on the mode.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            mode (str): Normalization mode. Supported values:
                - "max": Normalize by the maximum absolute value per channel.
                - "std": Normalize by the standard deviation per channel.
                - "": No normalization applied, returns data as is.

        Returns:
            np.ndarray: Normalized data array.

        Raises:
            ValueError: If an unsupported normalization mode is provided.
        """
        data -= np.mean(data, axis=1, keepdims=True)
        if mode == "max":
            max_data = np.max(data, axis=1, keepdims=True)
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == "std":
            std_data = np.std(data, axis=1, keepdims=True)
            std_data[std_data == 0] = 1
            data /= std_data
        elif mode == "":
            return data
        else:
            raise ValueError(f"Supported mode: 'max','std', got '{mode}'")
        return data

    def _generate_noise_data(self, data: np.ndarray, ppks: list, spks: list):
        """
        Remove all event phases from the data by replacing phase segments with noise (in-place).

        For each P-S phase pair, replaces the waveform segment from the P-phase pick
        to the estimated coda end (based on `coda_ratio`) with random Gaussian noise.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            ppks (list): List of P-phase pick indices.
            spks (list): List of S-phase pick indices.

        Returns:
            Tuple[np.ndarray, list, list]:
                - The modified data array with phases replaced by noise.
                - An empty list for P-phase picks (all removed).
                - An empty list for S-phase picks (all removed).
        """
        if len(ppks) > 0 and len(spks) > 0:
            for i in range(len(ppks)):
                ppk = ppks[i]
                spk = spks[i]
                coda_end = np.clip(
                    int(spk + self.coda_ratio * (spk - ppk)),
                    0,
                    data.shape[-1],
                    dtype=int,
                )
                if ppk < coda_end:
                    data[:, ppk:coda_end] = np.random.randn(
                        data.shape[0], coda_end - ppk
                    )

        return data, [], []

    def _add_event(self, data: np.ndarray, ppks: list, spks: list, min_gap: int):
        """
        Add a seismic event to the data by duplicating an existing event segment at a new location (in-place).

        Note:
            This method should be called before `_shift_event` to ensure proper augmentation.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            ppks (list): List of existing P-phase pick indices.
            spks (list): List of existing S-phase pick indices.
            min_gap (int): Minimum gap (in samples) to maintain between events to avoid overlap.

        Returns:
            Tuple[np.ndarray, list, list]:
                - Modified data array with the added seismic event.
                - Updated list of P-phase picks including the new event.
                - Updated list of S-phase picks including the new event.
        """
        target_idx = np.random.randint(0, len(ppks))

        ppk = ppks[target_idx]
        spk = spks[target_idx]
        coda_end = int(spk + (self.coda_ratio * (spk - ppk)))

        left = coda_end + min_gap
        right = data.shape[-1] - (spk - ppk) - min_gap

        if left < right:
            ppk_add = np.random.randint(left, right)
            spk_add = ppk_add + spk - ppk
            space = min(data.shape[-1] - ppk_add, coda_end - ppk)

            scale = np.random.random()

            data[:, ppk_add : ppk_add + space] += data[:, ppk : ppk + space] * scale

            ppks.append(ppk_add)
            spks.append(spk_add)

        ppks.sort()
        spks.sort()
        return data, ppks, spks

    def _shift_event(self, data, ppks, spks):
        """
        Shift the seismic events in the data circularly along the time axis.

        This method circularly shifts the data waveform and updates the P-phase and S-phase
        pick indices accordingly.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            ppks (list): List of P-phase pick indices.
            spks (list): List of S-phase pick indices.

        Returns:
            Tuple[np.ndarray, list, list]:
                - The circularly shifted data array.
                - Updated list of P-phase picks after the shift.
                - Updated list of S-phase picks after the shift.
        """
        shift = np.random.randint(0, data.shape[-1])
        data = np.concatenate((data[:, -shift:], data[:, :-shift]), axis=1)
        ppks = [(p + shift) % data.shape[-1] for p in ppks]
        spks = [(s + shift) % data.shape[-1] for s in spks]

        ppks.sort()
        spks.sort()
        return data, ppks, spks

    def _drop_channel(self, data):
        """
        Randomly drop (zero out) one or more channels in the input data (in-place).

        If the input data has fewer than 2 channels, no action is taken.
        Otherwise, a random number of channels between 1 and (number of channels - 1)
        are selected and set to zero.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).

        Returns:
            np.ndarray: The data array with randomly dropped channels zeroed out.
        """
        if data.shape[0] < 2:
            return data
        else:
            drop_num = np.random.choice(range(1, data.shape[0]))
            candidates = list(range(data.shape[0]))
            for _ in range(drop_num):
                c = np.random.choice(candidates)
                candidates.remove(c)
                data[c, :] = 0.0
        return data

    def _adjust_amplitude(self, data):
        """
        Adjust the amplitude of the input data after dropping channels. (inplace)

        The method scales the amplitude of the data such that the maximum absolute
        amplitude of each channel becomes the same value.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).

        Returns:
            np.ndarray: The data array with adjusted amplitudes.
        """
        max_amp = np.max(np.abs(data), axis=1)

        if np.count_nonzero(max_amp) > 0:
            data *= data.shape[0] / np.count_nonzero(max_amp)

        return data

    def _scale_amplitude(self, data):
        """
        Scale the amplitude of the input data. (inplace)

        The method randomly scales the amplitude by a factor between 1 and 3
        (either multiplying or dividing the data).

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).

        Returns:
            np.ndarray: The data array with scaled amplitudes.
        """
        if np.random.uniform(0, 1) < 0.5:
            data *= np.random.uniform(1, 3)
        else:
            data /= np.random.uniform(1, 3)

        return data

    def _pre_emphasis(self, data: np.ndarray, pre_emphasis: float) -> np.ndarray:
        """
        Apply pre-emphasis to the input data. (inplace)

        The method applies a high-pass filter to the input data, enhancing high-frequency components.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            pre_emphasis (float): Pre-emphasis coefficient.

        Returns:
            np.ndarray: The data array with pre-emphasis applied.
        """
        for c in range(data.shape[0]):
            bpf = data[c, :]
            data[c, :] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data

    def _add_noise(self, data):
        """
        Add Gaussian noise to the input data. (inplace)

        The method generates Gaussian noise with a random signal-to-noise ratio (SNR) and
        adds it to each channel of the input data.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).

        Returns:
            np.ndarray: The data array with Gaussian noise added.
        """
        for c in range(data.shape[0]):
            x = data[c, :]
            snr = np.random.randint(10, 50)
            px = np.sum(x**2) / len(x)
            pn = px * 10 ** (-snr / 10.0)
            noise = np.random.randn(len(x)) * np.sqrt(pn)
            data[c, :] += noise

        return data

    def _add_gaps(self, data: np.ndarray, ppks: list, spks: list):
        """
        Add gaps (zeros) in the input data. (inplace)

        The method creates random gaps between phases (P-wave, S-wave), setting portions
        of the data to zero. The gap locations are determined by the positions of the phases.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            ppks (list): List of P-wave phase indices.
            spks (list): List of S-wave phase indices.

        Returns:
            np.ndarray: The data array with gaps added.
        """
        phases = sorted(ppks + spks)

        if len(phases) > 0:
            phases.append(data.shape[-1] - 1)
            phases = sorted(set(phases))

            insert_pos = np.random.randint(0, len(phases) - 1)

            sgt = np.random.randint(phases[insert_pos], phases[insert_pos + 1])
            egt = np.random.randint(sgt, phases[insert_pos + 1])
        else:
            sgt = np.random.randint(0, data.shape[-1] - 1)
            egt = np.random.randint(sgt + 1, data.shape[-1])

        data[:, sgt:egt] = 0

        return data

    def _add_mask_windows(
        self,
        data: np.ndarray,
        percent: int = 50,
        window_size: int = 20,
        mask_value: float = 1.0,
    ):
        """
        Add mask windows to the input data. (inplace)

        This method adds mask windows, where a portion of the data is replaced with a specified
        mask value. The percentage of the data to be masked is controlled by the 'percent' argument.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            percent (int, optional): Percentage of windows to mask. Defaults to 50.
            window_size (int, optional): Size of each window to mask. Defaults to 20.
            mask_value (float, optional): Value to replace the window with. Defaults to 1.0.

        Returns:
            np.ndarray: The data array with masked windows.
        """
        p = np.clip(percent, 0, 100)
        num_windows = data.shape[-1] // window_size
        num_mask = num_windows * p // 100
        selected = np.random.choice(range(num_windows), num_mask, replace=False)
        for i in selected:
            st = i * window_size
            et = st + window_size
            data[:, st:et] = mask_value

        return data

    def _add_noise_windows(
        self, data: np.ndarray, percent: int = 50, window_size: int = 20
    ):
        """
        Add noise windows to the input data. (inplace)

        This method adds random noise to windows in the data. The percentage of the data to be
        corrupted with noise is controlled by the 'percent' argument.

        Args:
            data (np.ndarray): Input data array of shape (channels, samples).
            percent (int, optional): Percentage of windows to add noise. Defaults to 50.
            window_size (int, optional): Size of each window to add noise. Defaults to 20.

        Returns:
            np.ndarray: The data array with noise windows.
        """
        p = np.clip(percent, 0, 100)
        num_windows = data.shape[-1] // window_size
        num_block = num_windows * p // 100
        selected = np.random.choice(range(num_windows), num_block, replace=False)
        for i in selected:
            st = i * window_size
            et = st + window_size

            data[:, st:et] = np.random.randn(data.shape[0], window_size)

        return data

    def _data_augmentation(self, event: dict) -> dict:
        """Perform data augmentation on the input event.

        This function applies various data augmentation techniques on the event data, such as adding noise,
        scaling amplitude, shifting events, and dropping channels. It modifies the event in-place and
        returns the updated event dictionary.

        Args:
            event (dict): A dictionary containing the raw event data. Expected keys are:
                - "data": Raw data (e.g., waveform)
                - "ppks": List of P-phase indices
                - "spks": List of S-phase indices

        Returns:
            dict: The event dictionary with augmented data. The dictionary contains updated keys:
                - "data": Augmented waveform data
                - "ppks": Updated list of P-phase indices
                - "spks": Updated list of S-phase indices

        Notes:
            This method applies different augmentation techniques based on predefined probabilities for each
            transformation. Some transformations include noise generation, event addition, and data scaling.
            Additionally, window masking and noise windows are applied if their respective percentages are greater than 0.
        """
        data, ppks, spks = itemgetter("data", "ppks", "spks")(event)

        # Generate noise data
        if np.random.random() < self.generate_noise_rate:
            # Noise data
            data, ppks, spks = self._generate_noise_data(data, ppks, spks)
            self._clear_dict_except(event, "data")

            # Drop channel
            if np.random.random() < self.drop_channel_rate:
                data = self._drop_channel(data)
                data = self._adjust_amplitude(data)

            # Scale
            if np.random.random() < self.scale_amplitude_rate:
                data = self._scale_amplitude(data)

        else:
            # Add event
            for _ in range(self._max_event_num - len(ppks)):
                if np.random.random() < self.add_event_rate and ppks:
                    data, ppks, spks = self._add_event(
                        data, ppks, spks, self.min_event_gap
                    )

            # Shift event
            if np.random.random() < self.shift_event_rate:
                data, ppks, spks = self._shift_event(data, ppks, spks)

            # Drop channel
            if np.random.random() < self.drop_channel_rate:
                data = self._drop_channel(data)
                data = self._adjust_amplitude(data)

            # Scale
            if np.random.random() < self.scale_amplitude_rate:
                data = self._scale_amplitude(data)

            # Pre-emphasis
            if np.random.random() < self.pre_emphasis_rate:
                data = self._pre_emphasis(data, self.pre_emphasis_ratio)

            # Add noise
            if np.random.random() < self.add_noise_rate:
                data = self._add_noise(data)

            # Add gaps
            if np.random.random() < self.add_gap_rate:
                data = self._add_gaps(data, ppks, spks)

        if self.mask_percent > 0:
            data = self._add_mask_windows(
                data=data,
                percent=self.mask_percent,
                window_size=self.sampling_rate // 2,
            )

        if self.noise_percent > 0:
            data = self._add_noise_windows(
                data=data,
                percent=self.noise_percent,
                window_size=self.sampling_rate // 2,
            )

        event.update({"data": data, "ppks": ppks, "spks": spks})

        return event

    def process(self, event: dict, augmentation: bool, inplace: bool = True) -> dict:
        """Process raw data and apply optional augmentation.

        This function processes the raw event data, performs data augmentation if specified, and applies
        window cutting and normalization.

        Args:
            event (dict): A dictionary containing the raw event data. Expected keys are:
                - "data": Raw data (e.g., waveform)
                - "ppks": List of P-phase indices
                - "spks": List of S-phase indices
                - "snr": Signal-to-noise ratio of the event
            augmentation (bool): Whether to apply data augmentation to the event data.
            inplace (bool, optional): Whether to modify the event dictionary in place. Defaults to True.

        Returns:
            dict: The processed event data, which contains the following keys:
                - "data": Processed and normalized waveform data
                - "ppks": Updated list of P-phase indices
                - "spks": Updated list of S-phase indices

        Notes:
            The processing involves several steps like phase padding, event shifting, window cutting,
            and data normalization. If augmentation is enabled, additional transformations will be applied
            to the data.
        """
        if not inplace:
            event = copy.deepcopy(event)

        is_noise = self._is_noise(
            data=event["data"], ppks=event["ppks"], spks=event["spks"], snr=event["snr"]
        )

        # Noise
        if is_noise:
            self._clear_dict_except(event, "data")

        event["ppks"], event["spks"] = _pad_phases(
            event["ppks"], event["spks"], self.min_event_gap, self.in_samples
        )

        # Data augmentation
        if augmentation:
            event = self._data_augmentation(event=event)

        # Cut window
        event["data"], event["ppks"], event["spks"] = self._cut_window(
            data=event["data"],
            ppks=event["ppks"],
            spks=event["spks"],
            window_size=self.in_samples,
        )

        # Instance Norm
        event["data"] = self._normalize(event["data"], self.norm_mode)

        return event

    def _generate_soft_label(
        self, name: str, event: dict, soft_label_width: int, soft_label_shape: str
    ) -> np.ndarray:
        """Generate a soft label for a specific item in the event.

        This function generates a soft label (i.e., a label with a soft transition) for a given event item
        based on its name and specified label shape. The label can be in the form of a Gaussian,
        triangle, box, or sigmoid.

        Args:
            name (str): The name of the item to generate the soft label for.
                        Supported values: "ppk", "spk", "det", "non", "ppk+", "spk+", or a data channel name.
            event (dict): A dictionary containing the event data. Expected keys are:
                - "data": Raw data (e.g., waveform)
                - "ppks": List of P-phase indices
                - "spks": List of S-phase indices
            soft_label_width (int): The width of the soft label (i.e., the window size around each index).
            soft_label_shape (str): The shape of the soft label. Can be "gaussian", "triangle", "box", or "sigmoid".

        Returns:
            np.ndarray: The generated soft label as a NumPy array.

        Raises:
            NotImplementedError: If the specified label shape or name is not supported.
        """
        length = event["data"].shape[-1]

        def _clip(x: int) -> int:
            """Clip the index to be within valid bounds.

            This function ensures that the index `x` is within the range [0, length), where `length` is the
            total length of the event data.

            Args:
                x (int): The index to be clipped.

            Returns:
                int: The clipped index, constrained within the valid range [0, length).
            """
            return min(max(x, 0), length)

        def _get_soft_label(idxs, length):
            """Generate the soft label for a given set of indices.

            This function generates a soft label array of zeros with smooth transitions around the given indices.
            The window function applied around each index depends on the specified `soft_label_shape`.

            Args:
                idxs (list): A list of indices where the soft label should be applied.
                length (int): The length of the soft label array.

            Returns:
                np.ndarray: A soft label array with smooth transitions around the indices.
            """
            slabel = np.zeros(length)

            if len(idxs) > 0:
                left = int(soft_label_width / 2)
                right = soft_label_width - left

                if soft_label_shape == "gaussian":
                    window = np.exp(-((np.arange(-left, right + 1)) ** 2) / (2 * 10**2))
                elif soft_label_shape == "triangle":
                    window = 1 - np.abs(
                        2 / soft_label_width * (np.arange(-left, right + 1))
                    )
                elif soft_label_shape == "box":
                    window = np.ones(soft_label_width + 1)

                elif soft_label_shape == "sigmoid":

                    def _sigmoid(x):
                        return 1 / (1 + np.exp(x))

                    l_l, l_r = -int(left / 2), left - int(left / 2)
                    r_l, r_r = -int(right / 2), right - int(right / 2)
                    x_l, x_r = -10 / left * np.arange(l_l, l_r), -10 / right * (
                        -1
                    ) * np.arange(r_l, r_r)
                    w_l, w_r = _sigmoid(x_l), _sigmoid(x_r)
                    window = np.concatenate((w_l, [1.0], w_r), axis=0)
                else:
                    raise NotImplementedError(
                        f"Unsupported label shape: '{soft_label_shape}'"
                    )

                for idx in idxs:
                    if idx < 0:
                        pass  # Out of range
                    elif idx - left < 0:
                        slabel[: idx + right + 1] += window[
                            soft_label_width + 1 - (idx + right + 1) :
                        ]
                    elif idx + right <= length - 1:
                        slabel[idx - left : idx + right + 1] += window
                    elif idx <= length - 1:
                        slabel[-(length - (idx - left)) :] += window[
                            : length - (idx - left)
                        ]
                    else:
                        pass  # Out of range

            return slabel

        ppks, spks = _pad_phases(
            ppks=event["ppks"],
            spks=event["spks"],
            padding_idx=soft_label_width,
            num_samples=length,
        )

        # Phase-P/S
        if name in ["ppk", "spk"]:
            key = {"ppk": "ppks", "spk": "spks"}.get(name)
            label = _get_soft_label(idxs=event[key], length=length)

        # None (=1-P(p)-P(s))
        elif name == "non":
            label = (
                np.ones(length)
                - _get_soft_label(idxs=ppks, length=length)
                - _get_soft_label(idxs=spks, length=length)
            )
            label[label < 0] = 0

        # Detection
        elif name == "det":
            label = np.zeros(length)

            assert len(ppks) == len(spks)

            for i in range(len(ppks)):
                ppk = ppks[i]
                spk = spks[i]
                dst = ppk
                det = int(spk + (self.coda_ratio * (spk - ppk)))
                label_i = _get_soft_label(idxs=[dst, det], length=length)
                label_i[_clip(dst) : _clip(det)] = 1.0
                label += label_i
            label[label > 1] = 1.0

        # Phase-P/S (plus)
        elif name in ["ppk+", "spk+"]:
            label = np.zeros(length)
            key = {"ppk+": "ppks", "spk+": "spks"}.get(name)
            phases = event[key]
            for i in range(len(phases)):
                st = phases[i]
                label_i = _get_soft_label(idxs=[st], length=length)
                label_i[_clip(st) :] = 1.0
                label += label_i / len(phases)

        # Waveform
        elif name in self.data_channels:
            ch_idx = self.data_channels.index(name)
            label = event["data"][ch_idx]

        # Diff
        elif name in [f"d{c}" for c in self.data_channels]:
            channel_data = event["data"][self.data_channels.index(name[-1])]
            label = np.zeros_like(channel_data)
            label[1:] = np.diff(channel_data)

        else:
            raise NotImplementedError(f"Unsupported label name: '{name}'")

        return label.astype(self.dtype)

    def _get_io_item(
        self,
        name: Union[str, tuple, list],
        event: dict,
        soft_label_width: int = None,
        soft_label_shape: str = None,
    ) -> Union[tuple, list, np.ndarray]:
        """Get the input/output (IO) item for the event.

        This function retrieves the specified item (e.g., waveform, phase, or label) from the event data.
        The item can be returned as a NumPy array, tuple, or list, depending on the provided name.

        Args:
            name (Union[str, tuple, list]): The name of the item to retrieve. Can be a string (for a single item),
                                            a tuple or list (for multiple items).
            event (dict): A dictionary containing the event data.
            soft_label_width (int, optional): The width of the soft label. Defaults to None.
            soft_label_shape (str, optional): The shape of the soft label (e.g., "gaussian", "triangle"). Defaults to None.

        Returns:
            Union[tuple, list, np.ndarray]: The retrieved item. This can be a tuple or list for multiple items,
                                            or a NumPy array for a single item.

        Raises:
            ValueError: If the specified name does not exist in the event data.
            NotImplementedError: If the specified name or type is unsupported.
        """

        if isinstance(name, (tuple, list)):
            children = [self._get_io_item(sub_name, event) for sub_name in name]
            item = np.array(children)
            return item

        else:
            if Config.get_type(name) == "soft":
                item = self._generate_soft_label(
                    name=name,
                    event=event,
                    soft_label_width=(soft_label_width or self.soft_label_width),
                    soft_label_shape=(soft_label_shape or self.soft_label_shape),
                )

            elif Config.get_type(name) == "value":
                value = event[name]
                item = np.array(value).astype(self.dtype)

            elif Config.get_type(name) == "onehot":
                cidx = event[name]
                if not len(cidx) > 0:
                    raise ValueError(f"Item:{name}, Value:{cidx}")
                nc = Config.get_num_classes(name=name)
                item = np.eye(nc)[cidx[0]].astype(np.int64)

            else:
                raise NotImplementedError(f"Unknown item: {name}")

            return item

    def get_targets_for_loss(self, event: dict, label_names: list) -> Any:
        """Get the target values used to calculate the loss.

        This function retrieves the target values for a given list of label names. These targets are used
        during model training to compute the loss function.

        Args:
            event (dict): A dictionary containing the event data.
            label_names (list): A list of label names (e.g., "ppk", "spk", "det").

        Returns:
            Any: The targets corresponding to the given label names. Can be a tuple, list, or a single value.

        Notes:
            The method calls `_get_io_item` to retrieve each label and aggregates them into a tuple or list
            for the final output.
        """

        targets = [self._get_io_item(name=name, event=event) for name in label_names]

        if len(targets) > 1:
            return tuple(targets)
        else:
            return targets.pop()

    def get_targets_for_metrics(
        self,
        event: dict,
        max_event_num: int,
        task_names: list,
    ) -> dict:
        """Get labels used to calculate evaluation metrics.

        This function retrieves the labels necessary for calculating model evaluation metrics. It processes
        and pads the phases (P-phase and S-phase) to ensure they have the same length.

        Args:
            event (dict): A dictionary containing the event data.
            max_event_num (int): Maximum number of events to be considered for padding phase lists.
            task_names (list): A list of task names, which will determine the corresponding labels to be retrieved.

        Returns:
            dict: A dictionary of labels, where each key corresponds to a task name and each value is the
                corresponding label (e.g., "ppk", "spk", "det").

        Notes:
            The method applies padding and handles multiple types of label retrieval, including event detections.
        """

        targets = {}

        for name in task_names:
            if name in ["ppk", "spk"]:
                key = {"ppk": "ppks", "spk": "spks"}.get(name)
                tgt = self._get_io_item(name=key, event=event)
                tgt = _pad_array(
                    tgt, length=max_event_num, padding_value=int(-1e7)
                ).astype(np.int64)
            elif name == "det":
                padded_ppks, padded_spks = _pad_phases(
                    event["ppks"], event["spks"], self.soft_label_width, self.in_samples
                )
                detections = []
                for ppk, spk in zip(padded_ppks, padded_spks):
                    st = np.clip(ppk, 0, self.in_samples)
                    et = int(spk + (self.coda_ratio * (spk - ppk)))
                    detections.extend([st, et])
                expected_num = (
                    self._max_event_num
                    + int(bool(self.add_event_rate))
                    + int(bool(self.shift_event_rate))
                    + int(0 <= self.p_position_ratio <= 1)
                )
                if len(detections) // 2 < expected_num:
                    detections = detections + [1, 0] * (
                        expected_num - len(detections) // 2
                    )

                tgt = np.array(detections).astype(np.int64)

            else:
                tgt = self._get_io_item(name=name, event=event)

            targets[name] = tgt

        return targets

    def get_inputs(self, event: dict, input_names: list) -> Union[np.ndarray, tuple]:
        """Get input data for the model.

        This function retrieves the input data (e.g., waveform, phase) from the event, based on the specified
        list of input names. The inputs are returned as a NumPy array, tuple, or list.

        Args:
            event (dict): A dictionary containing the event data.
            input_names (list): A list of input names (e.g., "data", "ppks", "spks").

        Returns:
            Union[np.ndarray, tuple]: The input data as a NumPy array, tuple, or list, depending on the number of inputs.

        Notes:
            This function simplifies the retrieval of multiple inputs by checking each item name in the `input_names` list.
        """

        inputs = [self._get_io_item(name=name, event=event) for name in input_names]
        if len(inputs) > 1:
            return tuple(inputs)
        else:
            return inputs.pop()


class SeismicDataset(Dataset):
    """
    A dataset class for reading and preprocessing seismic data.

    This class handles loading, preprocessing, and augmentation of seismic data for training,
    validation, and testing tasks. It builds a dataset using the provided configuration and data
    preprocessing parameters. The class provides methods for retrieving preprocessed data, labels,
    and metadata.

    Attributes:
        _seed (int): Random seed for reproducibility.
        _mode (str): Mode of the dataset ('train', 'val', or 'test').
        _input_names (list): List of input names for the model.
        _label_names (list): List of label names.
        _task_names (list): List of task names for the model.
        _max_event_num (int): Maximum number of events in the dataset.
        _augmentation (bool): Whether data augmentation is applied.
        _dataset (Dataset): The dataset object containing the seismic events.
        _dataset_size (int): The size of the dataset.
        _preprocessor (DataPreprocessor): Preprocessor used for data transformation.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration options.
        input_names (list): List of input names. Refer to :class:`~SeisT.config.Config` for details.
        label_names (list): List of label names. Refer to :class:`~SeisT.config.Config` for details.
        task_names (list): List of task names. Refer to :class:`~SeisT.config.Config` for details.
        mode (str): Mode of the dataset. Can be 'train', 'val', or 'test'.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_names: list,
        label_names: list,
        task_names: list,
        mode: str,
    ) -> None:
        """
        Initializes the SeismicDataset object.

        Args:
            args (argparse.Namespace): Input arguments containing configuration options.
            input_names (list): List of input names. Refer to :class:`~SeisT.config.Config` for details.
            label_names (list): List of label names. Refer to :class:`~SeisT.config.Config` for details.
            task_names (list): List of task names. Refer to :class:`~SeisT.config.Config` for details.
            mode (str): Mode of the dataset. Can be 'train', 'val', or 'test'.
        """
        self._seed = int(args.seed)
        self._mode = mode.lower()
        self._input_names = input_names
        self._label_names = label_names
        self._task_names = task_names
        self._max_event_num = args.max_event_num

        self._augmentation = args.augmentation and self._mode == "train"
        if self._augmentation != args.augmentation:
            logger.warning(f"[{self._mode}]Augmentation -> {self._augmentation}")

        # Dataset
        self._dataset = build_dataset(
            dataset_name=args.dataset_name,
            seed=self._seed,
            mode=self._mode,
            data_dir=args.data,
            shuffle=args.shuffle,
            data_split=args.data_split,
            train_size=args.train_size,
            val_size=args.val_size,
        )
        logger.info(self._dataset)

        self._dataset_size = len(self._dataset)

        if self._augmentation:
            logger.warning(
                f"Data augmentation: Dataset size -> {self._dataset_size *2}"
            )

        # Preprocessor
        self._preprocessor = DataPreprocessor(
            data_channels=self._dataset.channels(),
            sampling_rate=self._dataset.sampling_rate(),
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * self._dataset.sampling_rate()),
            dtype=np.float32,
        )

    def sampling_rate(self):
        """
        Returns the sampling rate of the dataset.

        Returns:
            float: The sampling rate of the dataset.
        """
        return self._dataset.sampling_rate()

    def data_channels(self):
        """
        Returns the list of data channels in the dataset.

        Returns:
            list: A list of data channel names.
        """
        return self._dataset.channels()

    def name(self):
        """
        Returns the name of the dataset with its mode.

        Returns:
            str: The name of the dataset, including the mode (e.g., "train" or "test").
        """
        return f"{self._dataset.name()}_{self._mode}"

    def __len__(self) -> int:
        """
        Returns the size of the dataset, considering augmentation.

        If augmentation is enabled, the dataset size will be doubled (each event is duplicated with
        augmentation). Otherwise, the dataset size remains unchanged.

        Returns:
            int: The size of the dataset.
        """
        if self._augmentation:
            return 2 * self._dataset_size
        else:
            return self._dataset_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Fetches the data at the specified index.

        Args:
            idx (int): Index of the data item to fetch.

        Returns:
            tuple: A tuple containing the following elements:
                - inputs (Any): The preprocessed input data.
                - loss_targets (Any): The target labels for loss calculation.
                - metrics_targets (Any): The target labels for metrics calculation.
                - meta_data_json (str): The metadata in JSON format.
        """
        # Load data
        event, meta_data = self._dataset[idx % self._dataset_size]

        # Preprocess
        event = self._preprocessor.process(
            event=event, augmentation=(self._augmentation and idx >= self._dataset_size)
        )

        # Generate inputs
        inputs = self._preprocessor.get_inputs(
            event=event, input_names=self._input_names
        )

        # Generate labels
        loss_targets = self._preprocessor.get_targets_for_loss(
            event=event, label_names=self._label_names
        )
        metrics_targets = self._preprocessor.get_targets_for_metrics(
            event=event, task_names=self._task_names, max_event_num=self._max_event_num
        )
        meta_data_json = json.dumps(meta_data)
        return inputs, loss_targets, metrics_targets, meta_data_json
