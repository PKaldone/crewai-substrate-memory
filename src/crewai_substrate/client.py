"""
Synchronous HTTP client for the SUBSTRATE MCP server (JSON-RPC over HTTP).

The SUBSTRATE MCP endpoint accepts standard JSON-RPC 2.0 requests with
Bearer token authentication. This client wraps that transport into a
clean Python interface for use by the memory provider.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("crewai_substrate.client")

_DEFAULT_BASE_URL = "https://substrate.garmolabs.com/mcp-server/mcp"
_DEFAULT_TIMEOUT = 30.0
_JSONRPC_VERSION = "2.0"


class SubstrateClientError(Exception):
    """Raised when a SUBSTRATE MCP request fails."""

    def __init__(self, message: str, code: int = -1) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SubstrateClientConfig:
    """Immutable configuration for the SUBSTRATE MCP client."""

    api_key: str
    base_url: str = _DEFAULT_BASE_URL
    timeout: float = _DEFAULT_TIMEOUT


class SubstrateClient:
    """
    Synchronous JSON-RPC client for the SUBSTRATE MCP server.

    All tool calls go through ``call_tool`` which handles the JSON-RPC
    envelope, authentication, error handling, and response unwrapping.
    """

    def __init__(self, config: SubstrateClientConfig) -> None:
        if not config.api_key:
            raise ValueError("api_key is required -- set SUBSTRATE_API_KEY")
        self._config = config
        self._request_id = 0
        self._http = httpx.Client(
            base_url="",  # We use full URL in requests
            timeout=config.timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
        )
        self._session_id: str | None = None

    # -- Public API --------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """
        Call a SUBSTRATE MCP tool by name and return the result content.

        Raises ``SubstrateClientError`` on JSON-RPC errors or HTTP failures.
        """
        self._request_id += 1
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }
        return self._send(payload)

    def initialize(self) -> dict[str, Any]:
        """Perform the MCP initialize handshake."""
        self._request_id += 1
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": self._request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "crewai-substrate-memory",
                    "version": "0.1.0",
                },
            },
        }
        return self._send(payload)

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._http.close()

    # -- Internal ----------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> Any:
        """Send a JSON-RPC request and return the unwrapped result."""
        headers: dict[str, str] = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        try:
            response = self._http.post(
                self._config.base_url,
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise SubstrateClientError(f"HTTP request failed: {exc}") from exc

        # Capture session ID from response
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id

        if response.status_code == 401:
            raise SubstrateClientError("Authentication failed -- check your SUBSTRATE_API_KEY", code=-32000)

        if response.status_code == 429:
            raise SubstrateClientError("Rate limit exceeded -- slow down or upgrade your plan", code=-32029)

        if response.status_code not in (200, 202):
            raise SubstrateClientError(
                f"Unexpected HTTP {response.status_code}: {response.text[:200]}",
                code=response.status_code,
            )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise SubstrateClientError(f"Invalid JSON response: {exc}") from exc

        # Handle JSON-RPC error
        if "error" in body:
            err = body["error"]
            raise SubstrateClientError(
                err.get("message", "Unknown error"),
                code=err.get("code", -1),
            )

        return body.get("result", {})

    def __enter__(self) -> SubstrateClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
