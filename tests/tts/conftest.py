from app.tts.config import TTSSettings, get_tts_settings

# Tests do not read the developer's .env (see tests/conftest.py).
TTSSettings.model_config["env_file"] = None
get_tts_settings.cache_clear()
