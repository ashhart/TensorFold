"""SmartTensor exception types."""


class SmartTensorError(Exception):
    """Base class for SmartTensor errors."""


class UnsupportedModelFormatError(SmartTensorError):
    """Raised when a model format is not supported by the current runtime."""


class InvalidSafeTensorError(SmartTensorError):
    """Raised when a safetensors file is malformed or unsupported."""


class TensorNotFoundError(SmartTensorError, KeyError):
    """Raised when a tensor name is not present in the runtime."""
