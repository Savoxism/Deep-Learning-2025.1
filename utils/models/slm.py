import os
import torch
from dataclasses import dataclass
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


@dataclass
class SLMConfig:
    """
    Configuration for the Small Language Model (SLM).
    """
    # Using Llama-3-8B-Instruct
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct" 
    adapter_model: Optional[str] = None 
    
    load_in_4bit: bool = True
    device_map: str = "auto"
    max_seq_length: int = 2048


class LegalSLM:
    """
    Wrapper for Llama-3-Instruct using Transformers + BitsAndBytes + PEFT.
    """
    def __init__(self, config: SLMConfig = SLMConfig()):
        self.config = config

        # 1. Setup 4-bit Quantization
        bnb_config = None
        if self.config.load_in_4bit:
            print("⚙️ [LegalSLM] 4-bit quantization enabled.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # 2. Load Base Model
        print(f"🔄 [LegalSLM] Loading Base Model: {self.config.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            torch_dtype=torch.float16,   
            trust_remote_code=True,
        )

        # 3. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            use_fast=True,
        )

        # --- FIX: MANUALLY SET LLAMA 3 CHAT TEMPLATE ---
        # If the model card doesn't provide it automatically, we define it here.
        if self.tokenizer.chat_template is None:
            print("⚠️ [LegalSLM] Chat template missing. Injecting Llama 3 template manually.")
            self.tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}"
                "{% set content = '<|begin_of_text|>' + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )

        # Llama 3 padding fix
        if self.tokenizer.pad_token is None:
             self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4. Load Adapter
        if self.config.adapter_model:
            print(f"🧬 [LegalSLM] Attempting to load Adapter: {self.config.adapter_model}")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    self.config.adapter_model
                )
                print(f"✅ [LegalSLM] Adapter loaded successfully.")
            except Exception as e:
                print(f"❌ [LegalSLM] Failed to load adapter '{self.config.adapter_model}'.")
                print(f"   Error details: {e}")
                print("   ⚠️ Running with Base Model only.")

        self.model.eval()
        print("✅ [LegalSLM] Model initialization complete!")

    def _build_messages(self, context: str, question: str, task: str):
        """
        Build chat messages for Llama 3.
        """
        q_lower = (question or "").lower()

        # Determine if task is risk extraction
        is_risk = (task == "risk") or ("risk" in q_lower) or ("json" in q_lower)

        if is_risk:
            system = (
                "You are an expert legal auditor specializing in contract risk analysis. "
                "Analyze the provided clause and extract potential risks. "
                "Output your response strictly in valid JSON format."
            )
            user = f"Contract Clause:\n{context}"
        else:
            system = (
                "You are a helpful and precise legal assistant. "
                "Answer the user's question based strictly on the provided context below. "
                "If the information is not present in the context, clearly state that you do not know."
            )
            user = f"Context:\n{context}\n\nQuestion:\n{question}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, context: str, question: str, task: str = "qa") -> str:
        """
        Generate response based on context and question using Llama 3 template.
        """
        messages = self._build_messages(context=context, question=question, task=task)

        # Apply chat template (Now safe because we injected the template in __init__)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True 
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        # Define Llama 3 terminators
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators, 
            )

        # Remove the input prompt from the output
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return text


if __name__ == "__main__":
    print("--- Testing SLM (Llama 3 Config) ---")
    
    config = SLMConfig(
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_model=None 
    )

    try:
        # slm = LegalSLM(config)
        pass
    except Exception as e:
        print(f"Error: {e}")