import torch


def generate_response(model, tokenizer, user_input, memory_text=""):

    prompt = f"""
You are a helpful assistant.

User: {user_input}
Assistant:
"""

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # 🧹 anti bug DialoGPT
    if "you tried" in response.lower():
        return "Bonjour 👋 comment puis-je t'aider ?"

    return response if response else "Je ne comprends pas."
