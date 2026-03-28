"""Anthropic Agent SDK MCP tool server for BitPolar vector quantization.

Provides an MCP-compatible tool server with stdio and SSE transport
support for Claude Desktop integration and HTTP-based agent communication.
"""

from __future__ import annotations

import base64
import json
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar required. Install with: pip install bitpolar")


# MCP Tool Definitions — JSON-compatible descriptions for agent registration
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "bitpolar_compress",
        "description": (
            "Compress a float32 embedding vector to compact bytes using "
            "near-optimal quantization. No training needed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Float32 embedding vector to compress",
                },
                "bits": {
                    "type": "integer",
                    "default": 4,
                    "description": (
                        "Quantization precision (3=max compression, "
                        "4=recommended, 8=near-lossless)"
                    ),
                },
            },
            "required": ["vector"],
        },
    },
    {
        "name": "bitpolar_search",
        "description": (
            "Search a compressed vector index for the most similar "
            "vectors to a query."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Float32 query vector",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of results to return",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "bitpolar_add_vector",
        "description": "Add a vector to the compressed search index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Unique vector ID"},
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Float32 vector to add",
                },
            },
            "required": ["id", "vector"],
        },
    },
    {
        "name": "bitpolar_index_stats",
        "description": (
            "Get statistics about the compressed vector index "
            "(count, memory, compression ratio)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class BitPolarMCPServer:
    """MCP tool server for BitPolar with stdio and SSE transport support.

    Extends the MCP tool pattern with proper transport layers for
    Claude Desktop (stdio) and HTTP-based (SSE) agent integrations.

    Args:
        dim: Vector dimension.
        bits: Quantization precision (3-8).
        seed: Random seed for reproducible quantization.
        name: Server name for MCP protocol identification.

    Example:
        >>> server = BitPolarMCPServer(dim=384, bits=4)
        >>> server.run_stdio()      # For Claude Desktop
        >>> server.run_sse(port=8080)  # For HTTP transport
    """

    def __init__(
        self,
        dim: int = 384,
        bits: int = 4,
        seed: int = 42,
        name: str = "bitpolar-mcp",
    ) -> None:
        if not (3 <= bits <= 8):
            raise ValueError(f"bits must be 3-8, got {bits}")
        self._dim = dim
        self._bits = bits
        self._seed = seed
        self._name = name
        self._quantizer: Optional[_bp.TurboQuantizer] = None
        self._index: dict[int, np.ndarray] = {}

    def _ensure_quantizer(self) -> _bp.TurboQuantizer:
        """Lazily initialize the quantizer.

        Returns:
            The initialized TurboQuantizer instance.
        """
        if self._quantizer is None:
            proj = max(self._dim // 4, 1)
            self._quantizer = _bp.TurboQuantizer(
                dim=self._dim,
                bits=self._bits,
                projections=proj,
                seed=self._seed,
            )
        return self._quantizer

    def handle_tool_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle an MCP tool call by dispatching to the appropriate handler.

        Args:
            tool_name: One of the registered tool names.
            arguments: Tool arguments dict.

        Returns:
            Result dict with the tool's output, or an error dict.
        """
        handlers: dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "bitpolar_compress": self._handle_compress,
            "bitpolar_search": self._handle_search,
            "bitpolar_add_vector": self._handle_add,
            "bitpolar_index_stats": self._handle_stats,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(arguments)
        except Exception as e:
            return {"error": str(e)}

    def _handle_compress(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a vector to BitPolar compact codes.

        Args:
            args: Must contain 'vector', optional 'bits'.

        Returns:
            Dict with compressed base64 code and size statistics.
        """
        vector = np.array(args["vector"], dtype=np.float32)
        bits = args.get("bits", self._bits)

        if len(vector) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vector)}"}

        if not (3 <= bits <= 8):
            return {"error": f"bits must be 3-8, got {bits}"}

        if bits != self._bits:
            proj = max(self._dim // 4, 1)
            quantizer = _bp.TurboQuantizer(
                dim=self._dim, bits=bits, projections=proj, seed=self._seed
            )
        else:
            quantizer = self._ensure_quantizer()

        code = quantizer.encode(vector)
        code_b64 = base64.b64encode(bytes(code)).decode("ascii")

        return {
            "compressed": code_b64,
            "original_bytes": len(vector) * 4,
            "compressed_bytes": len(code),
            "compression_ratio": round(len(vector) * 4 / max(len(code), 1), 2),
        }

    def _handle_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search the compressed index for similar vectors.

        Args:
            args: Must contain 'query', optional 'top_k'.

        Returns:
            Dict with ranked results and total searched count.
        """
        query = np.array(args["query"], dtype=np.float32)
        top_k = args.get("top_k", 5)

        if len(query) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(query)}"}

        quantizer = self._ensure_quantizer()

        scored: list[dict[str, Any]] = []
        for vid, code in self._index.items():
            score = quantizer.inner_product(code, query)
            scored.append({"id": vid, "score": round(float(score), 6)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"results": scored[:top_k], "total_searched": len(self._index)}

    def _handle_add(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add a vector to the compressed index.

        Args:
            args: Must contain 'id' and 'vector'.

        Returns:
            Dict with success status, id, and index size.
        """
        vid = args["id"]
        vector = np.array(args["vector"], dtype=np.float32)

        if len(vector) != self._dim:
            return {"error": f"Expected {self._dim} dims, got {len(vector)}"}

        quantizer = self._ensure_quantizer()
        code = quantizer.encode(vector)
        self._index[vid] = code
        return {"success": True, "id": vid, "index_size": len(self._index)}

    def _handle_stats(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        """Return statistics about the compressed vector index.

        Args:
            _args: Ignored (no parameters needed).

        Returns:
            Dict with vector count, dimension, bits, and size statistics.
        """
        total_bytes = sum(len(c) for c in self._index.values())
        original_bytes = len(self._index) * self._dim * 4
        return {
            "vector_count": len(self._index),
            "dimension": self._dim,
            "bits": self._bits,
            "compressed_bytes": total_bytes,
            "original_bytes": original_bytes,
            "compression_ratio": round(
                original_bytes / max(total_bytes, 1), 2
            ),
        }

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible tool definitions."""
        return TOOL_DEFINITIONS

    def _create_jsonrpc_response(
        self, id: Any, result: Any
    ) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 response envelope.

        Args:
            id: The request ID to echo back.
            result: The result payload.

        Returns:
            A JSON-RPC 2.0 response dict.
        """
        return {"jsonrpc": "2.0", "id": id, "result": result}

    def _create_jsonrpc_error(
        self, id: Any, code: int, message: str
    ) -> Dict[str, Any]:
        """Create a JSON-RPC 2.0 error response envelope.

        Args:
            id: The request ID to echo back.
            code: Error code.
            message: Error message.

        Returns:
            A JSON-RPC 2.0 error response dict.
        """
        return {
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }

    def _handle_jsonrpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single JSON-RPC request for the MCP protocol.

        Handles initialize, tools/list, tools/call, and ping methods.

        Args:
            request: Parsed JSON-RPC request dict.

        Returns:
            JSON-RPC response dict.
        """
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            return self._create_jsonrpc_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self._name, "version": "0.2.0"},
            })

        if method == "notifications/initialized":
            # No response needed for notifications
            return self._create_jsonrpc_response(req_id, {})

        if method == "tools/list":
            return self._create_jsonrpc_response(req_id, {
                "tools": self.tool_definitions,
            })

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self.handle_tool_call(tool_name, arguments)

            if "error" in result:
                return self._create_jsonrpc_response(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": True,
                })

            return self._create_jsonrpc_response(req_id, {
                "content": [{"type": "text", "text": json.dumps(result)}],
            })

        if method == "ping":
            return self._create_jsonrpc_response(req_id, {})

        return self._create_jsonrpc_error(req_id, -32601, f"Method not found: {method}")

    def run_stdio(self) -> None:
        """Run the MCP server over stdio transport.

        Reads JSON-RPC requests from stdin line by line and writes
        responses to stdout. Suitable for Claude Desktop integration.

        The server runs until stdin is closed or an interrupt is received.
        """
        sys.stderr.write(f"BitPolar MCP Server ({self._name}) running on stdio\n")
        sys.stderr.flush()

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    error_resp = self._create_jsonrpc_error(
                        None, -32700, "Parse error"
                    )
                    sys.stdout.write(json.dumps(error_resp) + "\n")
                    sys.stdout.flush()
                    continue

                response = self._handle_jsonrpc_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except KeyboardInterrupt:
            sys.stderr.write("BitPolar MCP Server shutting down\n")
            sys.stderr.flush()

    def run_sse(self, port: int = 8080, host: str = "0.0.0.0") -> None:
        """Run the MCP server over HTTP with Server-Sent Events transport.

        Starts an HTTP server that accepts JSON-RPC requests via POST
        and streams responses via SSE. Suitable for remote agent integrations.

        Args:
            port: Port to listen on. Default 8080.
            host: Host to bind to. Default "0.0.0.0".

        Raises:
            ImportError: If the http.server module is not available (shouldn't happen).
        """
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        server_ref = self

        class MCPHandler(BaseHTTPRequestHandler):
            """HTTP request handler for MCP SSE transport."""

            def do_POST(self) -> None:
                """Handle POST requests containing JSON-RPC messages."""
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")

                try:
                    request = json.loads(body)
                except json.JSONDecodeError:
                    error_resp = server_ref._create_jsonrpc_error(
                        None, -32700, "Parse error"
                    )
                    self._send_json(400, error_resp)
                    return

                response = server_ref._handle_jsonrpc_request(request)
                self._send_json(200, response)

            def do_GET(self) -> None:
                """Handle GET requests for SSE event stream."""
                if self.path == "/sse":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header(
                        "Access-Control-Allow-Origin", "*"
                    )
                    self.end_headers()

                    # Send initial endpoint event
                    endpoint_data = json.dumps({
                        "endpoint": f"http://{host}:{port}/mcp",
                    })
                    self.wfile.write(
                        f"event: endpoint\ndata: {endpoint_data}\n\n".encode()
                    )
                    self.wfile.flush()

                    # Keep connection alive
                    try:
                        while True:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                            import time
                            time.sleep(30)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif self.path == "/health":
                    self._send_json(200, {"status": "ok", "server": server_ref._name})
                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_json(self, status: int, data: Any) -> None:
                """Send a JSON response.

                Args:
                    status: HTTP status code.
                    data: Data to serialize as JSON.
                """
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                """Suppress default logging to stderr."""
                pass

        httpd = HTTPServer((host, port), MCPHandler)
        sys.stderr.write(
            f"BitPolar MCP Server ({self._name}) running on http://{host}:{port}\n"
            f"  SSE endpoint: http://{host}:{port}/sse\n"
            f"  RPC endpoint: http://{host}:{port}/mcp (POST)\n"
            f"  Health check: http://{host}:{port}/health\n"
        )
        sys.stderr.flush()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            sys.stderr.write("BitPolar MCP Server shutting down\n")
            sys.stderr.flush()
            httpd.shutdown()
