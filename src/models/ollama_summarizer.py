from ollama import chat, pull


class OllamaSummarizer:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        output_keys: list[str],
        model_kwargs: dict = {},
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.output_keys = output_keys
        self.model_kwargs = model_kwargs
        respone = pull(model_name)
        print(respone)


    def __call__(self, text: str, shots: list[dict]=[]) -> dict:
        response = chat(
            self.model_name,
            [
                {"role": "system", "content": self.system_prompt},
                *shots,
                {"role": "user", "content": text},
            ],
            **self.model_kwargs,
        )
        return {key: response.get("message").get(key) for key in self.output_keys}
