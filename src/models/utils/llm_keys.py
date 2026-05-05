from src.utils import StrEnum


class LLMKeys(StrEnum):
    ROLE = "role"
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CONTENT = "content"
    THINKING = "thinking"