import time
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Make sure to use a CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the models you want to test
model_names = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]

# Define the sequence lengths you want to test
sequence_lengths = [128, 256, 512, 1024, 2047]

num_iterations = 100

# Initialize the data frame
df = pd.DataFrame(columns=["num_params", "seq_len", "time", "memory"])
CACHE_DIR = '/data/imoshkov/huggingface/'
# For each model
for model_name in model_names:
    # Load the model and the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)

    # Get the number of parameters
    num_params = sum(p.numel() for p in model.parameters())

    # For each sequence length
    for seq_len in sequence_lengths:
        times = []
        memories = []
        for _ in range(num_iterations):
            # Generate a random sequence of the desired length
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len)).to(device)
            
            # Record the initial memory usage
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            torch.cuda.synchronize()

            # Record the initial time
            time_before = time.time()

            # Perform a forward pass through the model
            outputs = model(input_ids)

            # Record the final time
            torch.cuda.synchronize()
            time_after = time.time()

            # Record the final memory usage
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()

            # Calculate the time and memory usage
            times.append(time_after - time_before)
            memories.append(mem_after - mem_before)

        # Calculate the average time and memory usage
        avg_time = sum(times) / len(times)
        avg_memory = sum(memories) / len(memories)

        # Add the results to the data frame
        df_dictionary = pd.DataFrame([{
            "num_params": num_params,
            "seq_len": seq_len,
            "time": avg_time,
            "memory": avg_memory
        }])
        df = pd.concat([df, df_dictionary], ignore_index=True)


save_dir = 'plots'
sns.set_style('whitegrid')
# Plot the results
r1 = sns.relplot(data=df, x="num_params", y="time", hue="seq_len", size="seq_len", kind="line")
r2 = sns.relplot(data=df, x="num_params", y="memory", hue="seq_len", size="seq_len", kind="line")
# sns.jointplot(data=df, x="num_params", y="time", hue="seq_len", kind="kde")
# sns.jointplot(data=df, x="num_params", y="memory", hue="seq_len", kind="kde")
r1.savefig(f"{save_dir}/relplot_time.png")
r2.savefig(f"{save_dir}/relplot_memory.png")