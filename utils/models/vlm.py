import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    BitsAndBytesConfig
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

@dataclass
class VLMConfig:
    """
    Configuration for the Vision Language Model.
    """
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_model: Optional[str] = None  
    load_in_4bit: bool = True            
    device_map: str = "auto"
    
    default_dpi: int = 200
    default_max_new_tokens: int = 1024

class VisionLanguageModel:
    """
    Wrapper for Qwen2.5-VL with support for LoRA Adapters (Local or HF Hub) and 4-bit quantization.
    """
    def __init__(self, config: VLMConfig = VLMConfig()):
        self.config = config

        # 1. Load Processor
        print(f"🔧 [VLM] Loading processor: {self.config.base_model}")
        self.processor = AutoProcessor.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )

        # 2. Configure 4-bit Quantization
        quantization_config = None
        if self.config.load_in_4bit:
            print("⚙️ [VLM] Enabling 4-bit quantization (BitsAndBytes)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # 3. Load Base Model
        print(f"🔄 [VLM] Loading Base Model: {self.config.base_model}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.base_model,
            device_map=self.config.device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if self.config.load_in_4bit else "auto"
        )

        # 4. Load LoRA Adapter (Updated logic)
        if self.config.adapter_model:
            print(f"🧬 [VLM] Attempting to load Adapter: {self.config.adapter_model}")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    self.config.adapter_model
                )
                print(f"✅ [VLM] Adapter loaded successfully.")
            except Exception as e:
                print(f"❌ [VLM] Failed to load adapter '{self.config.adapter_model}'.")
                print(f"   Error details: {e}")
                print("   ⚠️ Running with Base Model only.")

        self.model.eval()
        print("✅ [VLM] Model initialization complete.")

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def pdf_to_pil_pages(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
        doc = fitz.open(pdf_path)
        pages: list[Image.Image] = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i in range(len(doc)):
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
        doc.close()
        return pages
    
    def page_to_markdown(self, page_img: Image.Image, page_num: int, verbose: bool = True) -> str:
        # Prompt explicitly tuned for document parsing
        user_text = (
            "Analyze this document page image and extract its content into clean Markdown format.\n"
            "Rules:\n"
            "1. Preserve all headings and structural hierarchy.\n"
            "2. Represent tables using Markdown table syntax.\n"
            "3. Do not describe the visual layout, just transcribe the content.\n"
            "4. If there are mathematical formulas, use LaTeX format.\n"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": page_img},
                {"type": "text", "text": user_text},
            ],
        }]

        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self._model_device()) for k, v in inputs.items()}

        if verbose: print(f"   --> Inferencing Page {page_num}...")
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.default_max_new_tokens
            )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        md = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        return md

    def pdf_to_markdown(self, pdf_path: str, output_md_path: str, verbose: bool = True) -> str:
        pages = self.pdf_to_pil_pages(pdf_path, dpi=self.config.default_dpi)
        if verbose: print(f"📄 Loaded {len(pages)} pages from {os.path.basename(pdf_path)}")

        chunks: list[str] = []
        
        iterator = enumerate(pages, start=1)
        if tqdm: iterator = tqdm(iterator, total=len(pages), desc="OCR Processing")

        for page_num, page_img in iterator:
            md = self.page_to_markdown(page_img, page_num, verbose=False)
            chunks.append(f"\n\n{md}\n")

        full_md = "\n\n---\n\n".join(chunks)
        
        os.makedirs(os.path.dirname(output_md_path) or ".", exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        if verbose: print(f"💾 Markdown saved to: {output_md_path}")
        return output_md_path

if __name__ == "__main__":
    # load with LoRA Adapter from Hugging Face
    print("--- Testing Hugging Face Adapter Loading ---")
    conf = VLMConfig(
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_model="Ewengc21/qwen_qlora_dl_project" 
    )
    
    try:
        # vlm = VisionLanguageModel(conf)
        pass 
    except Exception as e:
        print(f"Main Error: {e}")