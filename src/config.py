import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class AppConfig:
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    target_language: str = "vi"


def load_config() -> AppConfig:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Please add it to your environment or .env file.")
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    target_language = os.getenv("TARGET_LANGUAGE", "vi")
    return AppConfig(gemini_api_key=api_key, gemini_model=model, target_language=target_language)


