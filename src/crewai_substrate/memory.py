"""
SUBSTRATE Memory Provider for CrewAI.

Replaces CrewAI's default memory with SUBSTRATE's causal memory engine,
giving your crew persistent memory, emotional context, and identity
continuity across sessions.

This provider implements CrewAI's external memory interface and adds
SUBSTRATE-exclusive capabilities (emotion, trust, identity verification)
that no other memory provider offers.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from crewai_substrate.client import (
    SubstrateClient,
    SubstrateClientConfig,
    SubstrateClientError,
)

logger = logging.getLogger("crewai_substrate.memory")


def _extract_text(result: Any) -> str:
    """
    Extract text content from a SUBSTRATE MCP tool result.

    The MCP tools/call response wraps content in a ``content`` list of
    typed blocks. This helper pulls out all text blocks and joins them.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content", [])
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(parts)
        # Fallback: return the dict as JSON
        return json.dumps(result, indent=2)
    return str(result)


def _parse_json_safe(text: str) -> dict[str, Any]:
    """Parse JSON from text, returning an empty dict on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


@dataclass(frozen=True)
class SubstrateMemoryConfig:
    """
    Configuration for the SUBSTRATE memory provider.

    ``api_key`` defaults to the ``SUBSTRATE_API_KEY`` environment variable.
    ``base_url`` defaults to the production SUBSTRATE MCP endpoint.
    """

    api_key: str = ""
    base_url: str = "https://substrate.garmolabs.com/mcp-server/mcp"
    timeout: float = 30.0

    def __post_init__(self) -> None:
        # Resolve api_key from env if not provided directly
        if not self.api_key:
            key = os.environ.get("SUBSTRATE_API_KEY", "")
            # frozen=True requires object.__setattr__ for late binding
            object.__setattr__(self, "api_key", key)


class SubstrateMemoryProvider:
    """
    CrewAI-compatible external memory provider backed by SUBSTRATE.

    Implements the standard ``save`` / ``search`` / ``reset`` interface that
    CrewAI expects, plus SUBSTRATE-exclusive methods for emotional context
    and entity state inspection.

    Usage::

        from crewai import Crew
        from crewai_substrate import SubstrateMemoryProvider

        memory = SubstrateMemoryProvider(api_key="sk_sub_...")
        crew = Crew(
            agents=[...],
            tasks=[...],
            memory=True,
            memory_config={"provider": memory},
        )
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://substrate.garmolabs.com/mcp-server/mcp",
        timeout: float = 30.0,
    ) -> None:
        config = SubstrateMemoryConfig(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        if not config.api_key:
            raise ValueError(
                "SUBSTRATE API key required. Pass api_key= or set SUBSTRATE_API_KEY env var."
            )
        self._client = SubstrateClient(
            SubstrateClientConfig(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )
        )

    # -- CrewAI memory interface -------------------------------------------

    def save(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Store a memory through SUBSTRATE's ``respond`` tool.

        The entity processes the message through causal memory, values, and
        reflection layers. The returned text is the entity's response, which
        includes any extracted causal rules or insights.

        Args:
            content: The text content to store in memory.
            metadata: Optional metadata dict. If provided, it is serialized
                      and appended to the message for context.

        Returns:
            The entity's response text after processing the memory.
        """
        message = content
        if metadata:
            message = f"{content}\n\n[metadata: {json.dumps(metadata)}]"

        try:
            result = self._client.call_tool("respond", {"message": message})
            return _extract_text(result)
        except SubstrateClientError as exc:
            logger.error("Failed to save memory: %s", exc)
            raise

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Search SUBSTRATE memory using hybrid retrieval (semantic + keyword).

        Falls back to ``memory_search`` if ``hybrid_search`` is not available
        on the current tier.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            A list of result dicts, each with at least a ``text`` key.
        """
        try:
            result = self._client.call_tool(
                "hybrid_search",
                {"query": query, "top_k": limit},
            )
            text = _extract_text(result)
            parsed = _parse_json_safe(text)
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed["results"]
            return [{"text": text}]
        except SubstrateClientError as exc:
            # hybrid_search requires pro tier; fall back to memory_search
            if exc.code == -32000 or "not available" in str(exc).lower():
                logger.info("hybrid_search unavailable, falling back to memory_search")
                return self._fallback_search(query, limit)
            raise

    def reset(self) -> None:
        """
        No-op for SUBSTRATE.

        SUBSTRATE entities maintain persistent identity and memory across
        sessions by design. Calling reset is acknowledged but does not
        destroy memory -- the entity's continuity is preserved.
        """
        logger.info("reset() called -- SUBSTRATE memory is persistent; no data was cleared")

    # -- SUBSTRATE-exclusive methods ---------------------------------------

    def get_emotional_context(self) -> dict[str, Any]:
        """
        Get the entity's emotional state vector (UASV).

        Returns a dict with valence, arousal, dominance, and certainty
        dimensions. No other memory provider offers this capability.

        Returns:
            Dict with emotional dimensions and their current values.
        """
        try:
            result = self._client.call_tool("get_emotion_state")
            text = _extract_text(result)
            parsed = _parse_json_safe(text)
            return parsed if parsed else {"raw": text}
        except SubstrateClientError as exc:
            logger.error("Failed to get emotional context: %s", exc)
            return {"error": str(exc)}

    def get_entity_state(self) -> dict[str, Any]:
        """
        Get the entity's identity verification and trust state.

        Combines ``verify_identity`` (cryptographic continuity proof) with
        ``get_trust_state`` (trust scores, consistency ratings).

        Returns:
            Dict with ``identity`` and ``trust`` sub-dicts.
        """
        state: dict[str, Any] = {}

        try:
            identity_result = self._client.call_tool("verify_identity")
            identity_text = _extract_text(identity_result)
            state["identity"] = _parse_json_safe(identity_text) or {"raw": identity_text}
        except SubstrateClientError as exc:
            logger.error("Failed to verify identity: %s", exc)
            state["identity"] = {"error": str(exc)}

        try:
            trust_result = self._client.call_tool("get_trust_state")
            trust_text = _extract_text(trust_result)
            state["trust"] = _parse_json_safe(trust_text) or {"raw": trust_text}
        except SubstrateClientError as exc:
            # get_trust_state requires pro tier
            logger.warning("get_trust_state unavailable: %s", exc)
            state["trust"] = {"error": str(exc)}

        return state

    def get_memory_stats(self) -> dict[str, Any]:
        """
        Get causal memory statistics.

        Returns episode count, rule count, average rule probability,
        and high-confidence rule count.
        """
        try:
            result = self._client.call_tool("memory_stats")
            text = _extract_text(result)
            parsed = _parse_json_safe(text)
            return parsed if parsed else {"raw": text}
        except SubstrateClientError as exc:
            logger.error("Failed to get memory stats: %s", exc)
            return {"error": str(exc)}

    # -- Internal ----------------------------------------------------------

    def _fallback_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Use basic memory_search when hybrid_search is unavailable."""
        try:
            result = self._client.call_tool("memory_search", {"query": query})
            text = _extract_text(result)
            parsed = _parse_json_safe(text)
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed["results"][:limit]
            return [{"text": text}]
        except SubstrateClientError as exc:
            logger.error("Fallback memory_search also failed: %s", exc)
            return []

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> SubstrateMemoryProvider:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
