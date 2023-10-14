import gradio as gr
import glob
import torch
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
import uuid


model_path = "models/20231014-18-19"


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def process_image(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    image.save(f"data/{str(uuid.uuid4())}_{text}.png")
    return text.split("_")[0]


demo = gr.Interface(
    fn=process_image,
    inputs=gr.inputs.Image(type="pil"),
    outputs="text",
    examples=list(glob.glob("examples/*.png")),
)

demo.launch(share=True)
