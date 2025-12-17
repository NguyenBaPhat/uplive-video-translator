import re
from typing import List

import google.generativeai as genai

from .config import AppConfig


class GeminiClient:
    """Thin wrapper around the Gemini Python SDK for this case study."""

    def __init__(self, config: AppConfig) -> None:
        genai.configure(api_key=config.gemini_api_key)
        self._model = genai.GenerativeModel(config.gemini_model)
        self._target_language = config.target_language

    @property
    def target_language(self) -> str:
        return self._target_language

    def translate_lines(self, lines: List[str]) -> List[str]:
        """
        Translate a list of subtitle lines into the configured target language.
        Uses a detailed prompt engineered for high-quality, context-aware translation.
        """
        if not lines:
            return []

        # Prompt engineered for high-quality subtitle translation with clear constraints.
        system_prompt = f"""You are a professional subtitle translator. Translate the following English subtitle lines into {self._target_language}.

**Translation Guidelines:**
- Translate for natural meaning, not word-for-word. Use idiomatic {self._target_language} expressions.
- Maintain the original tone (formal/casual/technical) and emotional emphasis.
- Keep translations concise for subtitle reading speed.
- Preserve technical terms, brand names, and proper nouns as-is.
- Handle slang and cultural references naturally.

**Format Requirements (CRITICAL):**
- Output EXACTLY {len(lines)} numbered lines matching the input structure.
- Each line format: "N. [translation]" where N is the line number.
- Do NOT add explanations, merge lines, or split lines.
- Do NOT include any text outside the numbered translations.

**Example:**
Input: 1. Hello everyone. 2. Welcome back.
Output: 1. [Translation 1] 2. [Translation 2]

Now translate:"""

        numbered = "\n".join(f"{idx+1}. {line}" for idx, line in enumerate(lines))
        full_prompt = f"{system_prompt}\n\n{numbered}"

        response = self._model.generate_content(
            full_prompt, request_options={"timeout": 60}
        )
        text = response.text.strip()

        # Parse the response with robust error handling.
        translated: List[str] = []
        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            
            # Handle numbered format: "1. Translated text" or "1) Translated text"
            if "." in raw_line and raw_line[0].isdigit():
                parts = raw_line.split(".", 1)
                if len(parts) == 2:
                    translated.append(parts[1].strip())
                else:
                    translated.append(raw_line)
            elif ")" in raw_line and raw_line[0].isdigit():
                parts = raw_line.split(")", 1)
                if len(parts) == 2:
                    translated.append(parts[1].strip())
                else:
                    translated.append(raw_line)
            else:
                # Fallback: use the line as-is if it doesn't match expected format.
                translated.append(raw_line)

        # Validation: ensure we have the same number of translations as input lines.
        if len(translated) != len(lines):
            # If parsing failed, try a more lenient approach: split by any numbering pattern.
            # This is a fallback to handle edge cases where Gemini's format slightly differs.
            fallback_translated = []
            for line in text.splitlines():
                line = line.strip()
                if line and not line[0].isdigit():
                    # This might be a continuation or a line without numbering.
                    continue
                # Try to extract content after any numbering pattern.
                match = re.match(r'^\d+[\.\)]\s*(.+)', line)
                if match:
                    fallback_translated.append(match.group(1).strip())
                elif line and not any(c in line for c in "0123456789"):
                    # Pure text line without numbering.
                    fallback_translated.append(line)
            
            if len(fallback_translated) == len(lines):
                return fallback_translated
            # Last resort: return as-is split, even if count doesn't match.
            return [t for t in translated if t] or text.splitlines()
        
        return translated


