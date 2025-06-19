from transformers import AutoTokenizer, AutoModelForCausalLM
from Configs.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)
