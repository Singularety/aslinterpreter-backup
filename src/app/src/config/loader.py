"""Configuration module for loader."""

import os
import toml
from pathlib import Path
from core.settings import Settings
from config.config import DEFAULT_CONFIG
CONFIG_DIR = Path(__file__).parent
FILENAME = "config.dev.toml"
CONFIG_PATH = CONFIG_DIR / FILENAME
def loadSettings():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = toml.load(f)

        if not data:
            raise ValueError("Empty config")

        return Settings(**data)

    except Exception:
        # fallback to defaults
        default = DEFAULT_CONFIG

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            toml.dump(default, f)

        return Settings(**default)
