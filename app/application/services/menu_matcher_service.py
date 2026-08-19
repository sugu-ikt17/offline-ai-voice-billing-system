"""Menu Matcher Service.

Resolves each parsed order item (name + quantity) to a real menu entry
stored in the database, returning both matched and unmatched items.

Matching is intentionally isolated from the parser (no text-to-speech logic)
and from billing (no tax, no totals).  Its only job is:

    ParsedOrderItem  →  OrderItem  (with price and per-line subtotal)

Three-tier matching strategy (strongest to weakest signal):
  1. Exact name match — case-insensitive, leading/trailing whitespace ignored.
  2. Shared-word match — "dosa" resolves to "Masala Dosa" because they share
     the word "dosa".  Ties broken by shortest candidate (most specific).
  3. Fuzzy whole-string fallback — difflib catches minor STT typos on
     single-word items (e.g. "tee" → "Tea").

Items that pass no tier land in `MatchResult.unmatched_items` so the caller
can surface a warning instead of silently losing the line.
"""

import difflib

from app.domain.entities.order import MatchResult, OrderItem, ParsedOrderItem
from app.domain.interfaces.menu_matcher_interface import MenuMatcherInterface
from app.infrastructure.database.models.menu_item_model import MenuItemModel
from app.infrastructure.database.repositories.menu_repository import MenuRepository

# Minimum difflib similarity ratio (0–1) for a fuzzy match to be accepted.
# 0.6 rejects clearly unrelated words while tolerating minor STT transcription
# errors ("dhosa" → "dosa").
_FUZZY_CUTOFF = 0.6


class MenuMatcherService(MenuMatcherInterface):
    """Matches parsed order items against the shop's menu database.

    Args:
        menu_repository: Repository providing access to MenuItemModel records.

    Usage::

        matcher = MenuMatcherService(menu_repo)
        result  = matcher.match([{"item": "dosa", "quantity": 2},
                                 {"item": "pizza", "quantity": 1}])

        result.matched_items    # [OrderItem(name="Masala Dosa", ...)]
        result.unmatched_items  # ["pizza"]
    """

    def __init__(self, menu_repository: MenuRepository) -> None:
        self._repo = menu_repository

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, parsed_items: list[ParsedOrderItem]) -> MatchResult:
        """Resolve each parsed item to a menu entry.

        Args:
            parsed_items: Output of OrderParserService — list of
                          ``{"item": str, "quantity": int}`` dicts.

        Returns:
            A :class:`MatchResult` with two attributes:

            * ``matched_items``   – :class:`OrderItem` list.  Each entry has
              ``menu_item_id``, ``name``, ``unit_price``, ``quantity``, and a
              computed ``line_total`` property.  The ``to_dict()`` method on
              each item serialises to the Bill Generator wire format:
              ``{"menu_id", "name", "price", "quantity", "subtotal"}``.

            * ``unmatched_items`` – list of spoken item name strings that
              could not be resolved to any menu entry.
        """
        if not parsed_items:
            return MatchResult()

        menu_items = self._repo.get_all()
        if not menu_items:
            # Every parsed item is unmatched when the menu is empty.
            return MatchResult(
                unmatched_items=[p["item"] for p in parsed_items]
            )

        # Build a normalised lookup: lower-stripped name → MenuItemModel.
        name_lookup: dict[str, MenuItemModel] = {
            item.name.strip().lower(): item for item in menu_items
        }
        candidate_names = list(name_lookup.keys())

        matched:   list[OrderItem] = []
        unmatched: list[str]       = []

        for parsed in parsed_items:
            spoken = parsed["item"].strip().lower()
            menu_item = self._resolve(spoken, name_lookup, candidate_names)

            if menu_item is not None:
                matched.append(
                    OrderItem(
                        menu_item_id=menu_item.id,
                        name=menu_item.name,
                        quantity=parsed["quantity"],
                        unit_price=float(menu_item.price),
                    )
                )
            else:
                unmatched.append(parsed["item"])

        return MatchResult(matched_items=matched, unmatched_items=unmatched)

    # ------------------------------------------------------------------
    # Internal matching helpers
    # ------------------------------------------------------------------

    def _resolve(
        self,
        spoken: str,
        name_lookup: dict[str, MenuItemModel],
        candidate_names: list[str],
    ) -> MenuItemModel | None:
        """Try all three matching tiers and return the best result, or None."""
        return (
            self._exact_match(spoken, name_lookup)
            or self._shared_word_match(spoken, name_lookup, candidate_names)
            or self._fuzzy_match(spoken, name_lookup, candidate_names)
        )

    @staticmethod
    def _exact_match(
        spoken: str,
        name_lookup: dict[str, MenuItemModel],
    ) -> MenuItemModel | None:
        """Tier 1 — case-insensitive, whitespace-stripped exact comparison."""
        return name_lookup.get(spoken)

    @staticmethod
    def _shared_word_match(
        spoken: str,
        name_lookup: dict[str, MenuItemModel],
        candidate_names: list[str],
    ) -> MenuItemModel | None:
        """Tier 2 — pick the candidate that shares the most words with the
        spoken name.  Ties are broken by preferring the shorter candidate
        (the more specific entry).

        Example: spoken="dosa", candidates=["masala dosa", "plain dosa"]
          → both share one word; shorter candidate wins → "plain dosa".
        """
        spoken_words = set(spoken.split())
        best_candidate: str | None = None
        best_overlap   = 0

        for candidate in candidate_names:
            overlap = len(spoken_words & set(candidate.split()))
            if overlap > best_overlap or (
                overlap == best_overlap
                and overlap > 0
                and best_candidate is not None
                and len(candidate) < len(best_candidate)
            ):
                best_overlap = overlap
                best_candidate = candidate

        if best_candidate is not None and best_overlap > 0:
            return name_lookup[best_candidate]
        return None

    @staticmethod
    def _fuzzy_match(
        spoken: str,
        name_lookup: dict[str, MenuItemModel],
        candidate_names: list[str],
    ) -> MenuItemModel | None:
        """Tier 3 — difflib whole-string similarity for single-word typos.

        Only attempted when the spoken name is a single word (multi-word
        items with no word overlap are better left unmatched than guessed
        by fuzzy similarity).

        Safety rule: spoken word must be at least 70% of the candidate's
        length to be eligible for fuzzy scoring.  This stops short words
        (e.g. "dosa", 4 chars) from matching much-longer candidates
        (e.g. "samosa", 6 chars: 4/6 = 0.67 < 0.70 → blocked), while
        still allowing "tee" → "tea" (3/3 = 1.0 ≥ 0.70 → eligible).
        """
        if len(spoken.split()) > 1:
            return None

        _MIN_LEN_RATIO = 0.70  # spoken must be ≥ 70 % of candidate length
        eligible = [c for c in candidate_names if len(spoken) / len(c) >= _MIN_LEN_RATIO]
        if not eligible:
            return None

        matches = difflib.get_close_matches(spoken, eligible, n=1, cutoff=_FUZZY_CUTOFF)
        return name_lookup[matches[0]] if matches else None
