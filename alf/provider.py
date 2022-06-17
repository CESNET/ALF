from abc import ABC, abstractmethod


class Provider(ABC):
    """Abstract class for contexts. Singleton design.
    """
    @staticmethod
    @abstractmethod
    def create_context(context_type: str, **options) -> None:
        """Create context.

        Args:
            context_type (str): Context type

        Exception:
            ValueError: If context_type is not a string
        """
    @staticmethod
    @abstractmethod
    def get_context():
        """Get context.

        Returns:
            Context of any type.
        """
