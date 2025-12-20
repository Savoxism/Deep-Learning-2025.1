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
    # Qwen2.5 Instruct model
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    adapter_model: Optional[str] = None  # Path to LoRA adapter (if any)

    load_in_4bit: bool = True
    device_map: str = "auto"

    max_seq_length: int = 2048


class LegalSLM:
    """
    Wrapper for Qwen2.5-3B-Instruct using Transformers + BitsAndBytes + PEFT.
    """
    def __init__(self, config: SLMConfig = SLMConfig()):
        self.config = config

        # LoRA Quantization setup
        bnb_config = None
        if self.config.load_in_4bit:
            print("⚙️ [LegalSLM] 4-bit quantization enabled.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # base model
        print(f"🔄 [LegalSLM] Loading Base Model: {self.config.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            dtype=torch.float16,   
            trust_remote_code=True,
        )

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            use_fast=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4) Load Adapter (if provided and exists)
        if self.config.adapter_model:
            if os.path.exists(self.config.adapter_model):
                print(f"🔄 [LegalSLM] Loading Adapter: {self.config.adapter_model}")
                self.model = PeftModel.from_pretrained(self.model, self.config.adapter_model)
            else:
                print(
                    f"⚠️ WARNING: Adapter path '{self.config.adapter_model}' not found. "
                    "Using Base Model only."
                )

        self.model.eval()
        print("✅ [LegalSLM] Model loaded successfully!")

    def _build_messages(self, context: str, question: str, task: str):
        """
        Build chat messages for Qwen2.5 Instruct.
        """
        q_lower = (question or "").lower()

        # Determine if task is risk extraction
        is_risk = (task == "risk") or ("risk" in q_lower) or ("json" in q_lower)

        if is_risk:
            system = (
                "You are a legal due diligence expert. "
                "Analyze the clause and extract risks. "
                "Return ONLY valid JSON."
            )
            user = f"Contract clause:\n{context}"
        else:
            system = (
                "You are a virtual legal assistant. "
                "Answer the user's question STRICTLY based on the provided context. "
                "Make the answer concise and to the point. "
                "If the answer is not in the context, say you do not know."
            )
            user = f"Context:\n{context}\n\nUser question:\n{question}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, context: str, question: str, task: str = "qa") -> str:
        """
        Generate response based on context and question.
        """
        messages = self._build_messages(context=context, question=question, task=task)

        # Qwen Instruct expects chat formatting via template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # important: appends assistant turn
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128, # short, concise answers
                temperature=0.1,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return text


if __name__ == "__main__":
    # Example 1: Load Base Model Only
    # config = SLMConfig(base_model="Qwen/Qwen2.5-3B-Instruct", adapter_model=None)

    # Example 2: Load with Adapter
    config = SLMConfig(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_model="models/qwen25_legal_adapter",
    )

    try:
        slm = LegalSLM(config)

        ctx = "The tenant must pay rent by the 5th of every month. Late fees are 5%."
        q = "When is the rent due?"
        print("\nAnswer:", slm.generate(ctx, q))

        # Risk/JSON example
        clause = "The supplier may terminate this agreement at any time with no notice."
        print("\nRisks:", slm.generate(clause, "Extract risks in JSON.", task="risk"))

    except Exception as e:
        print(f"Error during test: {e}")
