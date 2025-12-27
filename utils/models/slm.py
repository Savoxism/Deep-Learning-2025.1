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
        )
        
        # Padding fix for Llama 3
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
        
        # --- DEFINING TEMPLATE STRING HERE (Safe & Explicit) ---
        self.llama3_template = (
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
        
        print("✅ [LegalSLM] Model initialization complete!")

    def _build_messages(self, context: str, question: str, task: str):
        q_lower = (question or "").lower()
        is_risk = (task == "risk") or ("risk" in q_lower) or ("json" in q_lower)

        if is_risk:
            # Với tác vụ trích xuất rủi ro, vẫn cần nghiêm túc và chính xác
            system = (
                "You are an expert legal auditor. "
                "Analyze the clause below and extract potential risks into a JSON format. "
                "Be concise and objective."
            )
            user = f"Clause:\n{context}"
        else:
            system = (
                "You are a helpful and smart legal assistant. "
                "Your goal is to answer the user's question clearly and concisely based on the context provided. "
                "Avoid unnecessary legal jargon; explain simply if needed. "
                "If the answer is not in the context, politely say you don't have that information."
            )
            user = f"Context:\n{context}\n\nQuestion:\n{question}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, context: str, question: str, task: str = "qa") -> str:
        """
        Generate response passing the template explicitly.
        """
        messages = self._build_messages(context=context, question=question, task=task)

        # --- FIX: Pass 'chat_template' argument explicitly ---
        # This overrides whatever is inside the tokenizer configuration.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.llama3_template  # <--- HERE IS THE FIX
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators, 
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return text


if __name__ == "__main__":
    print("--- Testing SLM (Explicit Template) ---")
    config = SLMConfig(
        base_model="meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_model=None 
    )

    try:
        slm = LegalSLM(config)
        # Test nhanh để chắc chắn không lỗi template
        print(slm.generate("Rent is 500.", "What is rent?"))
    except Exception as e:
        print(f"Error: {e}")