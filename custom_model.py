import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, MarianMTModel, MarianTokenizer
from tqdm import tqdm

model_name = 'EleutherAI/gpt-neo-2.7B'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

translation_model_name = 'Helsinki-NLP/opus-mt-en-ru'
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)


def generate_comment(article, max_length=200, num_beams=5, temperature=0.7, top_k=50, top_p=0.95):
    prompts = [
        "Write a comment to the article expressing surprise:",
        "Write a comment to the article expressing outrage:",
        "Write a comment to the article expressing joy:"
    ]

    comments = []

    for prompt in tqdm(prompts, desc="Generating comments"):
        full_prompt = f"{prompt} {article}"
        input_ids = tokenizer.encode(full_prompt, return_tensors='pt')

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=2,
                early_stopping=True,
                do_sample=True
            )

        generated_comment = tokenizer.decode(output[0], skip_special_tokens=True)
        comments.append(generated_comment)

    return comments


def translate_comments(comments):
    translated_comments = []
    for comment in tqdm(comments, desc="Translating comments"):
        translated = translation_model.generate(
            **translation_tokenizer.prepare_seq2seq_batch([comment], return_tensors="pt"))
        translated_text = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translated_comments.append(translated_text[0])
    return translated_comments


article = "The transfer of 12.9 billion rubles abroad for the remaining IKEA goods in Russia was recognized by the court as 'immoral' and 'antisocial'."
print("Generating comments...")
generated_comments = generate_comment(article, max_length=200, num_beams=5, temperature=0.7, top_k=50, top_p=0.95)
print("Translating comments...")
translated_comments = translate_comments(generated_comments)

for i, comment in enumerate(translated_comments):
    print(f"Generated Comment {i + 1}:\n{comment}\n")
