from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "GenAI Deep Research Assistant"
    knowledge_base_dir: str = "data/knowledge_base"
    default_top_k: int = 4
    llm_provider: str = "local_demo"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
