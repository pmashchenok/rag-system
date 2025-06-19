from src.query_service.services.llm import model, tokenizer
from src.utils.logger import logging
from Configs.config import Config


def generate_answer(context_chunks, question):
    """Генерация ответа от LLM с учетом контекста"""
    context = "\n".join(context_chunks)
    prompt = f"Контекст:\n{context}\nВопрос: {question}"
    logging.info(f"Формируем ответ на основе промпта:\n{prompt[:200]}...")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=Config.MAX_INPUT_LENGTH
    ).to("cpu")
    input_len = inputs["input_ids"].shape[1]

    max_new_tokens = min(Config.MAX_NEW_TOKENS, tokenizer.model_max_length - input_len)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = answer.replace(prompt, "").strip()

    logging.info(f"Ответ LLM: {final_answer[:200]}...")
    return final_answer

