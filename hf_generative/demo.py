import gradio as gr
from transformers import AutoTokenizer
from ensemble_model import DecoderEnsemble

# Load your pre-trained language model and tokenizer
model_names = [
    'togethercomputer/RedPajama-INCITE-Base-3B-v1',
    'databricks/dolly-v2-3b',
    'stabilityai/stablelm-base-alpha-3b',
    # 'EleutherAI/pythia-2.8b-deduped'
    # 'gpt2', 'gpt2-medium'
]

# Create an instance of the ensemble model class
devices = ['cuda:1', 'cuda:2', 'cuda:0',]
model = DecoderEnsemble(model_names, devices)
tokenizer = AutoTokenizer.from_pretrained(model_names[0])

def generate_text(prompt, model_choices):
    model.set_models(model_choices)
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


# Define Gradio input and output components
input_text = gr.inputs.Textbox(lines=3, label="Input Prompt")
model_choice = gr.inputs.CheckboxGroup(choices=model_names, label="Select Models")
output_text = gr.outputs.Textbox(label="Generated Text")

# Create the Gradio interface
interface = gr.Interface(fn=generate_text, inputs=[input_text, model_choice], outputs=output_text, title="Language Model")

# Launch the interface (by default, it will be available at http://127.0.0.1:7860/)
interface.launch(share=True)
