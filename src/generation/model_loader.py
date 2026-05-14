import multiprocessing
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

from src.utils.logger import logger
from src.generation.config import GENERATION_MODEL_ID, GENERATION_ONNX_FILE, HF_TOKEN

MODEL_STORE = Path("model_store")
MODEL_PATH = MODEL_STORE / GENERATION_MODEL_ID.replace("/", "_")


def load_model():

    sess_options = ort.SessionOptions()
    cores = max(1, multiprocessing.cpu_count() - 1)
    sess_options.intra_op_num_threads = cores
    sess_options.inter_op_num_threads = cores
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if MODEL_PATH.exists():
        logger.info(f"Loading generation model from {MODEL_PATH}")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

        model = ORTModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            file_name=GENERATION_ONNX_FILE,
            use_cache=True,
            session_options=sess_options
        )
        return tokenizer, model

    logger.info(f"Downloading {GENERATION_MODEL_ID} ...")
    MODEL_STORE.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_ID, token=HF_TOKEN)
    model = ORTModelForCausalLM.from_pretrained(
        GENERATION_MODEL_ID,
        subfolder="onnx",
        file_name=GENERATION_ONNX_FILE,
        token=HF_TOKEN,
        use_cache=True,
        session_options=sess_options
    )

    tokenizer.save_pretrained(str(MODEL_PATH))
    model.save_pretrained(str(MODEL_PATH))
    logger.info(f"Saved to {MODEL_PATH}")

    return tokenizer, model