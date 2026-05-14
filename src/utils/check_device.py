import onnxruntime as ort
from src.utils.logger import logger

def get_onnx_provider() -> str:
    """Returns CUDAExecutionProvider if GPU is available, else CPUExecutionProvider."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        logger.info("GPU detected. Using CUDAExecutionProvider.")
        return "CUDAExecutionProvider"
    
    logger.warning("GPU not detected or onnxruntime-gpu mismatch. Falling back to CPUExecutionProvider.")
    return "CPUExecutionProvider"


get_onnx_provider()