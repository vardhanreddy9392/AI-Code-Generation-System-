import streamlit as st
import torch
from transformers import pipeline

# Handling randomness
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seed for reproducibility
set_seed(42)  # You can change the seed value as needed

# Load the model 
pipe = pipeline("text-generation", model="GuillenLuis03/PyCodeGPT")

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("", ["Home", "Code Generation"])

    if selection == "Home":
        show_home()
    elif selection == "Code Generation":
        show_code_generation()

def show_home():
    st.title("AI Code Generation Application")
    st.image("1.png", use_column_width=True)

def show_code_generation():
    st.title("AI Code Generation")

    # User input for prompt
    prompt = st.text_input("Enter a prompt to generate Python code:", "")

    if prompt:
        # Generate code
        generated_code = pipe(prompt, 
                              max_length=100, 
                              temperature=0.7, 
                              num_return_sequences=1
                              )[0]['generated_text']
        
        st.subheader("Generated Python code:")
        st.code(generated_code)
        st.balloons()

if __name__ == "__main__":
    main()
