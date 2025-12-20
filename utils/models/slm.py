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
    base_model: str = "unsloth/llama-3-8b-bnb-4bit" 
    adapter_model: Optional[str] = None
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    device_map: str = "auto"

class LegalSLM:
    """
    Wrapper for Llama-3 (SLM) using Transformers + BitsAndBytes + PEFT.
    """
    def __init__(self, config: SLMConfig = SLMConfig()):
        self.config = config
        
        # 1. Configure 4-bit Quantization (BitsAndBytes)
        bnb_config = None
        if self.config.load_in_4bit:
            print("⚙️ [LegalSLM] 4-bit quantization enabled.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )

        # 2. Load Base Model
        print(f"🔄 [LegalSLM] Loading Base Model: {self.config.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map=self.config.device_map,
            dtype=torch.float16,
            trust_remote_code=True,
        )

        # 3. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        # Ensure padding token exists (Llama-3 often requires this fix)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4. Load Adapter (if provided and exists)
        if self.config.adapter_model:
            if os.path.exists(self.config.adapter_model):
                print(f"🔄 [LegalSLM] Loading Adapter: {self.config.adapter_model}")
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    self.config.adapter_model
                )
            else:
                print(f"⚠️ WARNING: Adapter path '{self.config.adapter_model}' not found. Using Base Model only.")

        self.model.eval()
        
        # System prompt (Alpaca format)
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        """
        print(f"✅ [LegalSLM] Model loaded successfully!")

    def generate(self, context: str, question: str, task: str = "qa") -> str:
        """
        Generate response based on context and question.
        """
        # Select instruction based on task
        if task == "risk" or "risk" in question.lower() or "json" in question.lower():
            instruction = (
                "You are a legal due diligence expert. "
                "Analyze the following clause and extract the risks in JSON format."
            )
            full_input = f"Contract clause:\n{context}"
        else:
            instruction = (
                "You are a virtual legal assistant. "
                "Answer the user's question STRICTLY based on the information provided in the context below. "
                "If the information is not available in the context, say that you do not know."
            )
            full_input = f"Context:\n{context}\n\nUser question:\n{question}"

        # Format Prompt
        prompt = self.alpaca_prompt.format(instruction, full_input, "")

        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_seq_length
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                use_cache=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True # Set True for temperature to take effect
            )

        # Decode
        # Slicing [0][len(inputs["input_ids"][0]):] removes the prompt from the output
        generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
        clean_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Fallback cleanup just in case
        if "### Response:" in clean_response:
             clean_response = clean_response.split("### Response:\n")[1].strip()

        return clean_response

if __name__ == "__main__":
    # Example 1: Load Base Model Only
    # config = SLMConfig(base_model="unsloth/llama-3-8b-bnb-4bit", adapter_model=None)
    
    # Example 2: Load with Adapter
    config = SLMConfig(
        base_model="unsloth/llama-3-8b-bnb-4bit",
        adapter_model="models/llama3_legal_adapter" 
    )

    try:
        slm = LegalSLM(config)
        
        # Test Generation
        ctx = "The tenant must pay rent by the 5th of every month. Late fees are 5%."
        q = "When is the rent due?"
        print("\nAnswer:", slm.generate(ctx, q))
    except Exception as e:
        print(f"Error during test: {e}")