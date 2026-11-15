//! Unlimited-OCR-2's chat template, served when the checkpoint ships none (the release has no template).
pub const model_type = "unlimited-ocr";
pub const source = @embedFile("chat_template.jinja");
