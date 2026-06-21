import torch

def generate_response(model, tokenizer, user_input, memory_text=""):

    input_text = memory_text + f"User: {user_input}\nBot:"

    inputs = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        inputs,
        max_length=200,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

    return response
