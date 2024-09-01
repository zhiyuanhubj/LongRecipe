import torch
import logging
from utils.logger import TqdmToLogger
from transformers import AutoModelForCausalLM, AutoTokenizer





def load_model_and_tokenizer(model_path, accelerator):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=accelerator.device,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast_tokenizer=True)
    
    return model, tokenizer
    

def load_logger(log_path):
    
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    
    return logger, tqdm_out
