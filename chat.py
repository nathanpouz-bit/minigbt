import torch


def generate_response(model, tokenizer, user_input, memory_text=""):

    prompt = f"Human: {user_input}\nAssistant:"

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=150,
            do_sample=True,
            temperature=0.6,   # 🔥 plus stable
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,  # 🔥 IMPORTANT
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

    # 🧹 nettoyage réponses cassées
    response = response.strip()

    # 🔥 filtre anti-bizarre
    bad_outputs = ["you tried", "you tried.", "you tried to"]
    if any(bad in response.lower() for bad in bad_outputs):
        return "Bonjour 👋 ! Comment puis-je t'aider ?"

    return response if response else "Je ne comprends pas."
