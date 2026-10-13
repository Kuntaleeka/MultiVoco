from app.agent.config import AgentSettings, get_agent_settings
from tests.loan_fixtures import loan_db  # noqa: F401  (a fixture)

# Tests do not read the developer's .env (see tests/conftest.py).
AgentSettings.model_config["env_file"] = None
get_agent_settings.cache_clear()
