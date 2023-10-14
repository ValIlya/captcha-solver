import gradio as gr

import torch
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor

model_path = 'models/20231014-16-52'


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def process_image(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

demo = gr.Interface(
    fn=process_image,
    inputs=gr.inputs.Image(type="pil"),
    outputs="text",
    examples=['examples/c3xavu.png'],
)

demo.launch(share=True)
