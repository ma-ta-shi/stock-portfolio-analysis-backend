import ollama

MODEL = "qwen2.5:14b"

class BaseAgent:
    def __init__(self, system_prompt:str):
        self.message = [
            {"role": "system", "content": system_prompt}
        ]
    def run(self, prompt:str):
        self.message.append({"role": "user", "content": prompt})
        response = ollama.chat(
            model=MODEL, 
            messages=self.message
            )
        output = response["message"]["content"]
        self.message.append({"role": "assistant", "content": output})
        return output
    