import transformers as llm_pkg
from transformers import BitsAndBytesConfig
import torch
from .utils import LLMKeys
from copy import deepcopy


class HuggingFaceSummarizer(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        model_config: dict,
        tokenizer_name: str,
        tokenizer_config: dict,
        system_prompt: str,
        shots: dict[str, str] = dict(),
        quantization_config: dict = dict(),
        generation_kwargs: dict = dict(),
        tokenizer_kwargs: dict = dict(),
        chat_template_kwargs: dict = dict(),
        think: bool = False,
    ) -> None:
        super().__init__()
        bnb_config = (
            BitsAndBytesConfig(**quantization_config)
            if len(list(quantization_config.keys())) != 0
            else None
        )
        self.model = getattr(llm_pkg, model_name).from_pretrained(
            **model_config, quantization_config=bnb_config
        )
        self.model.eval()
        self.tokenizer = getattr(llm_pkg, tokenizer_name).from_pretrained(
            **tokenizer_config
        )
        self.tokenizer.add_special_tokens(
            {"extra_special_tokens": ["</think>", " assistant", "<think>"]}
        )
        self.system_prompt = system_prompt
        self.shots = shots
        self.generation_kwargs = generation_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.chat_template_kwargs = chat_template_kwargs
        self._init_system_messages()
        self.think = think

    def _init_system_messages(self) -> None:
        self.system_messages = [
            {LLMKeys.ROLE: LLMKeys.SYSTEM, LLMKeys.CONTENT: self.system_prompt}
        ]
        for user_input, assistant_response in self.shots.items():
            self.system_messages.append(
                {LLMKeys.ROLE: LLMKeys.USER, LLMKeys.CONTENT: user_input}
            )
            self.system_messages.append(
                {LLMKeys.ROLE: LLMKeys.ASSISTANT, LLMKeys.CONTENT: assistant_response}
            )

    def build_prompt(self, user_input: str) -> list[dict[str, str]]:
        prompt = deepcopy(self.system_messages)
        prompt.append({LLMKeys.ROLE: LLMKeys.USER, LLMKeys.CONTENT: user_input})
        return self.tokenizer.apply_chat_template(prompt, **self.chat_template_kwargs)

    def divide_think_by_content(
        self,
        response: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        think_id = self.tokenizer.convert_tokens_to_ids("</think>")
        mask = response == think_id
        has_token = mask.any(dim=1)

        first_pos = mask.float().argmax(dim=1)
        think_pos = torch.where(has_token, first_pos, torch.full_like(first_pos, -1))

        B, T = response.shape
        device = response.device
        pad_id = self.tokenizer.pad_token_id

        split_pos = torch.where(
            think_pos >= 0, think_pos, torch.full_like(think_pos, T)
        )

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        thinking_mask = positions < split_pos.unsqueeze(1)
        content_mask = positions > split_pos.unsqueeze(1)

        thinking = torch.where(thinking_mask, response, pad_id)
        content = torch.where(content_mask, response, pad_id)
        return thinking, content

    def delete_prompt_from_response(
        self, inputs, outputs: torch.LongTensor
    ) -> torch.LongTensor:

        B, T_total = outputs.shape
        device = outputs.device
        pad_id = self.tokenizer.pad_token_id

        # Full prompt width (including left padding)
        T_prompt = inputs["input_ids"].size(1)

        # Generation always starts after full prompt tensor
        generated = outputs[:, T_prompt:]

        return generated

    def delete_prompt_from_response_old(
        self, inputs, outputs: torch.LongTensor
    ) -> torch.LongTensor:
        attention_mask = inputs["attention_mask"]
        prompt_lens = attention_mask.sum(dim=1)

        B, T_total = outputs.shape
        device = outputs.device
        pad_id = self.tokenizer.pad_token_id

        gen_lengths = T_total - prompt_lens
        max_gen_len = gen_lengths.max()

        base_positions = (
            torch.arange(max_gen_len, device=device).unsqueeze(0).expand(B, -1)
        )
        gather_positions = base_positions + prompt_lens.unsqueeze(1)

        valid_mask = base_positions < gen_lengths.unsqueeze(1)

        gather_positions = torch.where(
            valid_mask, gather_positions, torch.zeros_like(gather_positions)
        )

        generated = torch.gather(outputs, 1, gather_positions)
        generated = torch.where(valid_mask, generated, pad_id)
        return generated

    @torch.no_grad()
    def forward(self, user_inputs: list[str]) -> list[str]:
        prompts = [self.build_prompt(user_input) for user_input in user_inputs]
        inputs = self.tokenizer(prompts, **self.tokenizer_kwargs).to(self.model.device)
        outputs = self.model.generate(**inputs, **self.generation_kwargs)
        generated = self.delete_prompt_from_response(inputs, outputs)
        if not self.think:
            return {
                LLMKeys.CONTENT: [
                    x.replace("`", "")
                    .strip()
                    .replace("assistant", "")
                    .strip("\n")
                    .replace("\\n", "\n")
                    for x in self.tokenizer.batch_decode(
                        generated, skip_special_tokens=True
                    )
                ]
            }
        thinking, content = self.divide_think_by_content(generated)
        thinking_text = self.tokenizer.batch_decode(thinking, skip_special_tokens=True)
        content_text = self.tokenizer.batch_decode(content, skip_special_tokens=True)
        thinking_text = [t.strip("\n") for t in thinking_text]
        content_text = [c.strip("\n") for c in content_text]

        return {LLMKeys.THINKING: thinking_text, LLMKeys.CONTENT: content_text}
