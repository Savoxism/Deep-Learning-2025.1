from utils.models.vlm import VLMConfig, VisionLanguageModel

cfg = VLMConfig(
    base_model="Qwen/Qwen2.5-VL-7B-Instruct",
    adapter_model="Ewengc21/qwen_qlora_dl_project",  # or None if no adapter
        device_map="auto",
        torch_dtype="auto",
        default_dpi=200,
        default_max_new_tokens=1024,
    )

vlm = VisionLanguageModel(config = cfg)

input_pdf = "data/sample.pdf"
output_md = "outputs/sample.md"

out_path = vlm.pdf_to_markdown(
    pdf_path=input_pdf,
    output_md_path=output_md,
    dpi = cfg.default_dpi,
    max_new_tokens=cfg.default_max_new_tokens,
    show_progress=True,
    verbose=True,
)

print(f"Converted Markdown saved to: {out_path}")