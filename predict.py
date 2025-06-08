from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class Predictor(BasePredictor):
    def setup(self):
        model_name = "mistralai/Mistral-7B-v0.1"
        adapter_path = "./"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        self.model = PeftModel.from_pretrained(base_model, adapter_path).eval()

    def predict(self, prompt: Input(str)) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output_ids = self.model.generate(**inputs, max_new_tokens=150)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)