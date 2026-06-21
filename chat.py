import torch

def generate_response(model, tokenizer, user_input):

    prompt = f"User: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Assistant:" in text:
        return text.split("Assistant:")[-1].strip()

    return text.strip()
