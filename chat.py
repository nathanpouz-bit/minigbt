import torch


def generate_response(model, tokenizer, user_input, memory_text=""):

    prompt = f"User: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # 🔥 IMPORTANT : extraire juste la réponse
    if "Assistant:" in decoded:
        response = decoded.split("Assistant:")[-1].strip()
    else:
        response = decoded.strip()

    return response if response else "Bonjour 👋"
