import dashscope
from dashscope import Generation

class QwenFallback:
    def __init__(self, api_key):
        dashscope.api_key = api_key
        self.model_name = "qwen-max"

    def get_response(self, query):
        prompt = f"""
You are an educational assistant chatbot named orBIT. You help students learn about undergraduate programs like IT, CS, and EMC.
Answer the following query clearly and accurately:

{query}
"""

        response = Generation.call(
            model=self.model_name,
            prompt=prompt,
            input_text=query
        )

        return response.output.text