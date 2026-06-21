import torch

def generate_response(model, tokenizer, user_input, memory_text=""):

    messages = [
        {
            "role": "user",
            "content": user_input
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return response.strip()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
