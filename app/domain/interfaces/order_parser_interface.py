"""Contract for converting raw transcribed text into structured order items."""

from abc import ABC, abstractmethod

from app.domain.entities.order import ParseResult, ParsedOrderItem


class OrderParserInterface(ABC):
    @abstractmethod
    def parse(self, text: str) -> list[ParsedOrderItem]:
        """Parse raw transcript text into a list of {"item", "quantity"} entries."""
        raise NotImplementedError

    @abstractmethod
    def parse_with_details(
        self, text: str, vocabulary: list[str] | None = None
    ) -> ParseResult:
        """Parse transcript text into detailed ParseResult (recognized, unknown, confidence)."""
        raise NotImplementedError
