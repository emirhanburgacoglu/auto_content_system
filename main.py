from core.text_generator import generate_motivational_quote, save_quote_to_file

def run():
    quote = generate_motivational_quote()
    print("[✔] Generated Quote:\n", quote)
    save_quote_to_file(quote)

if __name__ == "__main__":
    run()
