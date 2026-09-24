"""Abstract Translation Backend Interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    latency_ms: float
    is_success: bool = True
    error_message: str = ""

class BaseTranslator(ABC):
    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str = "en",
    ) -> TranslationResult:
        """Translate text from source_lang to target_lang."""
        pass
