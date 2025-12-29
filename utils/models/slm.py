import torch
from dataclasses import dataclass
from typing import Optional, List, Dict

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
    adapter_model: Optional[str] = None  # Path to your fine-tuned adapter (e.g., "./outputs/checkpoint-100")
    
    load_in_4bit: bool = True
    device_map: str = "auto"
    max_seq_length: int = 4096  # Llama 3 supports up to 8k, set 4k for safety on T4
    temperature: float = 0.3
    max_new_tokens: int = 256

class LegalSLM:
    """
    Wrapper for Llama-3-Instruct using Transformers + BitsAndBytes + PEFT (QLoRA).
    """
    def __init__(self, config: SLMConfig = SLMConfig()):
        self.config = config

        # 1. Setup 4-bit Quantization (NF4 - QLoRA standard)
        bnb_config = None
        if self.config.load_in_4bit:
            print(" [LegalSLM] 4-bit quantization (NF4) enabled.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # 2. Load Base Model
        print(f" [LegalSLM] Loading Base Model: {self.config.base_model}")
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

        # 4. Load Adapter (QLoRA)
        # Nếu có adapter_model, load weights đè lên base model
        if self.config.adapter_model:
            print(f" [LegalSLM] Attempting to load Adapter: {self.config.adapter_model}")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    self.config.adapter_model
                )
                print(f" [LegalSLM] Adapter loaded successfully.")
            except Exception as e:
                print(f" [LegalSLM] Failed to load adapter. Error: {e}")
                print("    Running with Base Model only.")

        self.model.eval()
        
        # Explicit Llama 3 Chat Template (Jinja2)
        # Đảm bảo format đúng chuẩn <|start_header_id|>...
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
        
        print(" [LegalSLM] Model initialization complete!")

    def _build_messages(self, context: str, question: str, task: str) -> List[Dict]:
        """
        Xây dựng prompt chuyên biệt để fix lỗi dài dòng và đảm bảo output đúng format.
        """
        q_lower = (question or "").lower()
        
        # Logic tự động phát hiện task nếu không truyền vào
        if task == "auto":
            if "risk" in q_lower or "json" in q_lower or "extract" in q_lower:
                task = "risk"
            else:
                task = "qa"

        if task == "risk":
            # --- TASK A: RISK EXTRACTION (JSON) ---
            system_prompt = (
                "You are a Senior Legal Auditor. Your task is to analyze the contract clause provided below. "
                "Output ONLY a valid JSON object containing the following keys: "
                "'risk_level' (High/Medium/Low), 'flagged_issues' (list of strings), and 'recommendation' (string). "
                "Do not add any markdown formatting like ```json. Do not explain your answer."
            )
            user_content = f"Clause:\n{context}"
            
        else:
            # --- TASK B: CITATION-AWARE QA ---
            system_prompt = (
                "You are a precise Legal Assistant. Answer the user's question based ONLY on the context provided. "
                "Rules:\n"
                "1. Be concise and direct.\n"
                "2. You MUST cite the specific section or clause number (e.g., 'According to Section 2.1...').\n"
                "3. If the answer is not in the context, state 'Information not found in the provided clause.'\n"
                "4. DO NOT copy the entire context. DO NOT start with 'Here is the summary'."
            )
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def generate(self, context: str, question: str = "", task: str = "auto") -> str:
        """
        Generate response with explicit templating.
        """
        messages = self._build_messages(context=context, question=question, task=task)

        # Apply template manually
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self.llama3_template
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
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators, 
            )

        # Cắt bỏ phần prompt đầu vào, chỉ lấy phần model sinh ra
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return text

# --- TESTING BLOCK ---
if __name__ == "_main_":
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