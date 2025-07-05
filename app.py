import torch
from transformers import pipeline
# handling randomness
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# Load the model 
pipe = pipeline("text-generation", model="GuillenLuis03/PyCodeGPT")
# Example 1
prompt = input("Enter the prompt: ")
generated_code = pipe(prompt, 
                      max_length=28, 
                      temperature=0.7, 
                      num_return_sequences=1
                      )[0:]['generated_text']

print("Generated Python code-->")
print(generated_code) 
