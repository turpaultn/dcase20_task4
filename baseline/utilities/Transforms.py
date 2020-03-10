import librosa
import numpy as np
import torch


class Transform:
    def transform_data(self, data):
        # Mandatory to be defined by subclasses
        raise NotImplementedError("Abstract object")

    def transform_label(self, label):
        # Do nothing, to be changed in subclasses if needed
        return label

    def _apply_transform(self, sample_no_index):
        data, label = sample_no_index
        if type(data) is tuple:  # meaning there is more than one data_input (could be duet, triplet...)
            data = list(data)
            for k in range(len(data)):
                data[k] = self.transform_data(data[k])
            data = tuple(data)
        else:
            data = self.transform_data(data)
        label = self.transform_label(label)
        return data, label

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        if type(sample[1]) is int:  # Means there is an index, may be another way to make it cleaner
            sample_data, index = sample
            sample_data = self._apply_transform(sample_data)
            sample = sample_data, index
        else:
            sample = self._apply_transform(sample)
        return sample


class GaussianNoise(Transform):
    """ Apply gaussian noise
        Args:
            mean: float, the mean of the gaussian distribution.
            std: float, standard deviation of the gaussian distribution.
        Attributes:
            mean: float, the mean of the gaussian distribution.
            std: float, standard deviation of the gaussian distribution.
        """

    def __init__(self, mean=0, std=0.5):
        self.mean = mean
        self.std = std

    def transform_data(self, data):
        """ Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            np.array
            The transformed data
        """
        return data + np.abs(np.random.normal(0, 0.5 ** 2, data.shape))


class ApplyLog(Transform):
    """Convert ndarrays in sample to Tensors."""

    def transform_data(self, data):
        """ Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            np.array
            The transformed data
        """
        return librosa.amplitude_to_db(data.T).T


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.
    The sequence should be on axis -2.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    shape = x.shape
    if shape[-2] <= max_len:
        padded = max_len - shape[-2]
        padded_shape = ((0, 0),)*len(shape[:-2]) + ((0, padded), (0, 0))
        x = np.pad(x, padded_shape, mode="constant")
    else:
        x = x[..., :max_len, :]
    return x


class PadOrTrunc(Transform):
    """ Pad or truncate a sequence given a number of frames
    Args:
        nb_frames: int, the number of frames to match
    Attributes:
        nb_frames: int, the number of frames to match
    """

    def __init__(self, nb_frames, apply_to_label=False):
        self.nb_frames = nb_frames
        self.apply_to_label = apply_to_label

    def transform_label(self, label):
        if self.apply_to_label:
            return pad_trunc_seq(label, self.nb_frames)
        else:
            return label

    def transform_data(self, data):
        """ Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            np.array
            The transformed data
        """
        return pad_trunc_seq(data, self.nb_frames)


class AugmentGaussianNoise(Transform):
    """ Pad or truncate a sequence given a number of frames
           Args:
               mean: float, mean of the Gaussian noise to add
           Attributes:
               std: float, std of the Gaussian noise to add
           """

    def __init__(self, mean=0, std=0.5):
        self.mean = mean
        self.std = std

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified

            Returns:
                (np.array, np.array)
                (original data, noise applied to original data)
        """
        noise = data + np.abs(np.random.normal(0, 0.5 ** 2, data.shape))
        return data, noise


class ToTensor(Transform):
    """Convert ndarrays in sample to Tensors.
    Args:
        unsqueeze_axis: int, (Default value = None) add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    Attributes:
        unsqueeze_axis: int, add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    """
    def __init__(self, unsqueeze_axis=None):
        self.unsqueeze_axis = unsqueeze_axis

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified

            Returns:
                np.array
                The transformed data
        """
        res_data = torch.from_numpy(data).float()
        if self.unsqueeze_axis is not None:
            res_data = res_data.unsqueeze(self.unsqueeze_axis)
        return res_data

    def transform_label(self, label):
        return torch.from_numpy(label).float()  # float otherwise error


class Normalize(Transform):
    """Normalize inputs
    Args:
        scaler: Scaler object, the scaler to be used to normalize the data
    Attributes:
        scaler : Scaler object, the scaler to be used to normalize the data
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified

            Returns:
                np.array
                The transformed data
        """
        return self.scaler.normalize(data)
