import os
from openai import OpenAI


class HindiRefiner:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def refine(self, text: str) -> str:
        prompt = f"""
                    You are a professional Hindi editor.

                    Rewrite the following Hindi sentence to sound natural, conversational,
                    and instructional. Do NOT change meaning.
                    Keep length roughly similar to original.

                    Original Hindi:
                    {text}

                    Improved Hindi:
                """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You improve Hindi translation quality."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        refined_text = response.choices[0].message.content.strip()
        return refined_text
