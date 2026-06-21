import torch

def generate_response(model, tokenizer, user_input):

    prompt = f"User: {user_input}\nBot:"

    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Bot:" in text:
        return text.split("Bot:")[-1].strip()

    return text.strip()
