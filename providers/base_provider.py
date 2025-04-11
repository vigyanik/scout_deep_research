# providers/base_provider.py

from abc import ABC, abstractmethod
from typing import Any, Type # Added Type
from config.settings import Settings, ModelProviderConfig
from models.schemas import SearchContent, BaseModel # Import specific schemas if needed for typing

class AgentProvider(ABC):
    """Abstract base class for all agent providers."""

    def __init__(self, settings: Settings, model_config: ModelProviderConfig):
        """
        Initializes the provider with global settings and specific model config.

        Args:
            settings: The application settings object.
            model_config: The specific model and provider configuration for this agent instance.
        """
        self.settings = settings
        self.model_config = model_config
        # Access agent-specific settings if available, requires AgentSettings in settings.py
        self.agent_settings = getattr(settings, 'agents', None)

    @abstractmethod
    def run(self, *args: Any, response_schema: Type[BaseModel] = None, **kwargs: Any) -> Any:
        """
        The main execution method for the agent.
        Must be implemented by subclasses. Input args and return type may vary.

        Args:
            *args: Positional arguments specific to the agent implementation.
            response_schema: Optional Pydantic model to validate/structure the response.
            **kwargs: Keyword arguments specific to the agent implementation.

        Returns:
            The result of the agent's execution, type depends on implementation.
        """
        pass

