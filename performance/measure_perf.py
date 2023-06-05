import time
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import gc

# Make sure to use a CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the models you want to test
model_names = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]

# Define the sequence lengths you want to test
sequence_lengths = [1, 128, 256, 512, 1024]

num_iterations = 100

# Initialize the data frame
df = pd.DataFrame(columns=["num_params", "seq_len", "time", "memory"])
CACHE_DIR = '/data3/imoshkov/huggingface/'
# For each model
for model_name in model_names:
    # Load the model and the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)

    # Get the number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # For each sequence length
    for seq_len in sequence_lengths:
        # Generate a random sequence of the desired length
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len)).to(device)
        
        # Record the initial memory usage
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        mem_before = torch.cuda.memory_allocated()
        torch.cuda.synchronize()

        # Record the initial time
        time_before = time.time()

        for _ in range(num_iterations):
            # Perform a forward pass through the model
            outputs = model(input_ids)

        # Record the final time
        torch.cuda.synchronize()
        time_after = time.time()
        mem_after = torch.cuda.memory_allocated()

        # Calculate the average time and memory usage
        avg_time = time_after - time_before
        avg_memory = mem_after - mem_before

        # Add the results to the data frame
        if seq_len == 1: continue

        df_dictionary = pd.DataFrame([{
            "num_params": num_params,
            "seq_len": seq_len,
            "time": avg_time,
            "memory": avg_memory / (1024 ** 2)
        }])
        df = pd.concat([df, df_dictionary], ignore_index=True)


print(df.to_json())
save_dir = 'plots'
sns.set_style('whitegrid')
# Plot the results
sns.set(font_scale=1.3)
sns.color_palette("rocket")

r1 = sns.relplot(data=df, x="num_params", y="time", hue="seq_len", size="seq_len", kind="line")
r2 = sns.relplot(data=df, x="num_params", y="memory", hue="seq_len", size="seq_len", kind="line")
sns.jointplot(data=df, x="num_params", y="time", hue="seq_len", kind="kde")
sns.jointplot(data=df, x="num_params", y="memory", hue="seq_len", kind="kde")
r1.savefig(f"{save_dir}/relplot_time.png")
r2.savefig(f"{save_dir}/relplot_memory.png")