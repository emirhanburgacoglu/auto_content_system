from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime

# Model ve tokenizer yükle
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Cihaz ayarı: GPU varsa cuda, yoksa cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_motivational_quote():
    prompt = "Write a short, powerful motivational quote in 1-2 sentences."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Prompt kısmını metinden temizle
    quote = text.replace(prompt, "").strip()
    return quote

def save_quote_to_file(quote):
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/contents/quote_{date_str}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(quote)
    print(f"[✓] Quote saved to {filename}")
