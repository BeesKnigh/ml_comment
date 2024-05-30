import asyncio
import fastapi_poe as fp

import config


class NeuroCommentator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pre_prompt = "Создай комментарии к статье. Вот название и первые 100 символов статьи:"

    async def generate_comments(self, article_title, article_start):
        prompt = f"{self.pre_prompt}\nНазвание статьи: {article_title}\nТекст статьи: {article_start}"
        message = fp.ProtocolMessage(role="user", content=prompt)

        full_text = ""
        async for partial in fp.get_bot_response(messages=[message], bot_name="GPT-3.5-Turbo", api_key=self.api_key):
            response_data = partial.dict()
            full_text += response_data.get('text', '')

        return full_text


api_key = config.API_TOKEN
neuro_commentator = NeuroCommentator(api_key)


article_title = "Дети в России тоже любят шоколад»: глава Ritter Sport объяснил, почему компания продолжает поставки в страну"
article_start = "Несмотря на «угрозы» Ронкен считает правильным решение продолжить работу с Россией и принял бы его снова, если бы пришлось."

comments = asyncio.run(neuro_commentator.generate_comments(article_title, article_start))
print(comments)