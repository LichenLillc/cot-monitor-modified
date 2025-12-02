from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "cais/HarmBench-Llama-2-13b-cls"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
