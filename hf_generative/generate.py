from transformers import GPT2LMHeadModel, AutoTokenizer, PreTrainedModel, PretrainedConfig

from ensemble_model import DecoderEnsemble

# Instantiate tokenizer
model_names = [
    'togethercomputer/RedPajama-INCITE-Base-3B-v1',
    'databricks/dolly-v2-3b'
    # 'gpt2', 'gpt2'
]
tokenizer = AutoTokenizer.from_pretrained(model_names[0])

# Define prompt
prompt = "Once upon a time, in a land far away"

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Create an instance of the ensemble model class
devices = ['cuda:0', 'cuda:1']
ensemble_model = DecoderEnsemble(model_names, devices)

# Generate text using the ensemble model
max_length = 100
num_return_sequences = 1

generated_sequences = ensemble_model.generate(
    input_ids, max_length=max_length,
    num_return_sequences=num_return_sequences
)

# Print the generated sequences
for i, seq in enumerate(generated_sequences):
    text = tokenizer.decode(seq, clean_up_tokenization_spaces=True)
    print(f"Generated Sequence {i+1}: {text}")
