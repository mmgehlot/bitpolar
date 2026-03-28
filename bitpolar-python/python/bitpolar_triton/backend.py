"""Triton Inference Server backend and client for BitPolar quantization.

Provides a Python backend implementing the Triton backend interface for
serving BitPolar quantization as an inference model, plus a client wrapper
for submitting compress/search requests to a running Triton server.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import bitpolar as _bp
except ImportError:
    raise ImportError("bitpolar is required. Install with: pip install bitpolar")


def _validate_bits(bits: int) -> None:
    """Validate quantization bit-width is in the supported range."""
    if not (3 <= bits <= 8):
        raise ValueError(f"bits must be 3-8, got {bits}")


class BitPolarTritonBackend:
    """Triton Python backend for BitPolar vector quantization.

    Implements the Triton Inference Server Python backend interface for
    serving BitPolar compress, decompress, and search operations as
    inference requests.

    Triton calls :meth:`initialize` once at model load, :meth:`execute`
    for each inference batch, and :meth:`finalize` at model unload.

    Args:
        dim: Vector dimension (default 384)
        bits: Quantization precision (3-8, default 4)
        projections: Number of QJL projections (default: dim/4)
        seed: Random seed

    Example (Triton model config):
        ```
        name: "bitpolar"
        backend: "python"
        parameters: {
            key: "dim", value: { string_value: "384" }
            key: "bits", value: { string_value: "4" }
        }
        ```
    """

    def __init__(
        self,
        dim: int = 384,
        bits: int = 4,
        projections: Optional[int] = None,
        seed: int = 42,
    ):
        _validate_bits(bits)
        self._dim = dim
        self._bits = bits
        self._projections = projections
        self._seed = seed
        self._quantizer: Optional["_bp.TurboQuantizer"] = None
        self._stored_codes: Dict[str, np.ndarray] = {}

    def initialize(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the backend — called once by Triton at model load.

        Reads configuration from args dict (Triton model parameters)
        and creates the TurboQuantizer.

        Args:
            args: Dict of model parameters from Triton config.
                  Recognized keys: "dim", "bits", "projections", "seed".
                  All values are strings (Triton convention).
        """
        if args:
            if "dim" in args:
                self._dim = int(args["dim"])
            if "bits" in args:
                self._bits = int(args["bits"])
                _validate_bits(self._bits)
            if "projections" in args:
                self._projections = int(args["projections"])
            if "seed" in args:
                self._seed = int(args["seed"])

        proj = self._projections or max(self._dim // 4, 1)
        self._quantizer = _bp.TurboQuantizer(
            dim=self._dim,
            bits=self._bits,
            projections=proj,
            seed=self._seed,
        )

    def _ensure_initialized(self) -> None:
        """Ensure the quantizer is initialized."""
        if self._quantizer is None:
            self.initialize()

    def execute(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process a batch of inference requests.

        Each request is a dict with an "operation" field indicating the
        action to perform:

        - **compress**: Compress vectors.
          Input: {"operation": "compress", "vectors": ndarray(n, dim)}
          Output: {"codes": list[ndarray], "dim": int, "bits": int}

        - **decompress**: Decompress codes.
          Input: {"operation": "decompress", "codes": list[ndarray]}
          Output: {"vectors": ndarray(n, dim)}

        - **inner_product**: Score codes against queries.
          Input: {"operation": "inner_product", "codes": list[ndarray],
                  "queries": ndarray(nq, dim)}
          Output: {"scores": ndarray(nq, n_codes)}

        - **search**: Top-K search over stored codes.
          Input: {"operation": "search", "queries": ndarray(nq, dim),
                  "codes": list[ndarray], "top_k": int}
          Output: {"indices": ndarray(nq, k), "scores": ndarray(nq, k)}

        - **store**: Store codes with IDs for later search.
          Input: {"operation": "store", "id": str, "vector": ndarray(dim,)}
          Output: {"status": "ok", "id": str}

        Args:
            requests: List of request dicts

        Returns:
            List of response dicts, one per request.
            Errors produce {"error": str} in the response.
        """
        self._ensure_initialized()
        assert self._quantizer is not None

        responses = []
        for req in requests:
            try:
                op = req.get("operation", "compress")
                resp = self._dispatch(op, req)
                responses.append(resp)
            except Exception as e:
                responses.append({"error": str(e)})

        return responses

    def _dispatch(
        self,
        operation: str,
        req: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route a request to the appropriate handler."""
        assert self._quantizer is not None

        if operation == "compress":
            return self._handle_compress(req)
        elif operation == "decompress":
            return self._handle_decompress(req)
        elif operation == "inner_product":
            return self._handle_inner_product(req)
        elif operation == "search":
            return self._handle_search(req)
        elif operation == "store":
            return self._handle_store(req)
        else:
            return {"error": f"Unknown operation: {operation}"}

    def _handle_compress(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Compress vectors to BitPolar codes."""
        assert self._quantizer is not None

        vectors = np.asarray(req["vectors"], dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n = vectors.shape[0]
        codes = []
        for i in range(n):
            codes.append(
                self._quantizer.encode(np.ascontiguousarray(vectors[i]))
            )

        return {
            "codes": codes,
            "dim": self._dim,
            "bits": self._bits,
        }

    def _handle_decompress(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress codes back to float32 vectors."""
        assert self._quantizer is not None

        codes = req["codes"]
        vectors = np.empty((len(codes), self._dim), dtype=np.float32)
        for i, code in enumerate(codes):
            vectors[i] = self._quantizer.decode(code)

        return {"vectors": vectors}

    def _handle_inner_product(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Compute inner products between codes and queries."""
        assert self._quantizer is not None

        codes = req["codes"]
        queries = np.asarray(req["queries"], dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        nq = queries.shape[0]
        nc = len(codes)
        scores = np.empty((nq, nc), dtype=np.float32)

        for qi in range(nq):
            for ci, code in enumerate(codes):
                scores[qi, ci] = self._quantizer.inner_product(
                    code, np.ascontiguousarray(queries[qi])
                )

        return {"scores": scores}

    def _handle_search(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Top-K search over provided or stored codes."""
        assert self._quantizer is not None

        queries = np.asarray(req["queries"], dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        codes = req.get("codes")
        if codes is None:
            # Use stored codes
            codes = list(self._stored_codes.values())

        top_k = int(req.get("top_k", 10))
        k = min(top_k, len(codes))

        if k == 0:
            return {
                "indices": np.empty((queries.shape[0], 0), dtype=np.int64),
                "scores": np.empty((queries.shape[0], 0), dtype=np.float32),
            }

        nq = queries.shape[0]
        indices = np.empty((nq, k), dtype=np.int64)
        scores_out = np.empty((nq, k), dtype=np.float32)

        for qi in range(nq):
            scores = np.empty(len(codes), dtype=np.float32)
            for ci, code in enumerate(codes):
                scores[ci] = self._quantizer.inner_product(
                    code, np.ascontiguousarray(queries[qi])
                )

            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            indices[qi] = top_idx
            scores_out[qi] = scores[top_idx]

        return {"indices": indices, "scores": scores_out}

    def _handle_store(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Store a compressed vector by ID."""
        assert self._quantizer is not None

        doc_id = str(req["id"])
        vector = np.asarray(req["vector"], dtype=np.float32)
        if vector.shape[0] != self._dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} != expected {self._dim}")
        code = self._quantizer.encode(np.ascontiguousarray(vector))
        self._stored_codes[doc_id] = code
        return {"status": "ok", "id": doc_id}

    def finalize(self) -> None:
        """Clean up resources — called by Triton at model unload."""
        self._quantizer = None
        self._stored_codes.clear()

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self._dim

    @property
    def bits(self) -> int:
        """Quantization bit-width."""
        return self._bits

    def __repr__(self) -> str:
        return (
            f"BitPolarTritonBackend(dim={self._dim}, bits={self._bits}, "
            f"stored={len(self._stored_codes)})"
        )


class BitPolarTritonClient:
    """Client for a BitPolar model deployed on Triton Inference Server.

    Wraps tritonclient to submit compress, decompress, and search
    requests to a running Triton server hosting a BitPolar Python backend.

    Falls back to HTTP if gRPC is unavailable.

    Args:
        url: Triton server URL (e.g. "localhost:8000" for HTTP,
             "localhost:8001" for gRPC)
        model_name: Name of the deployed BitPolar model
        protocol: "grpc" or "http" (default: "http")

    Example:
        >>> client = BitPolarTritonClient("localhost:8000", "bitpolar")
        >>> result = client.compress(np.random.randn(384).astype(np.float32))
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        protocol: str = "http",
    ):
        self._url = url
        self._model_name = model_name
        self._protocol = protocol
        self._client: Any = None
        self._infer_input_cls: Any = None
        self._infer_output_cls: Any = None

        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Triton client based on protocol."""
        if self._protocol == "grpc":
            try:
                import tritonclient.grpc as grpcclient

                self._client = grpcclient.InferenceServerClient(url=self._url)
                self._infer_input_cls = grpcclient.InferInput
                self._infer_output_cls = grpcclient.InferRequestedOutput
            except ImportError:
                raise ImportError(
                    "tritonclient[grpc] is required. "
                    "Install with: pip install tritonclient[grpc]"
                )
        else:
            try:
                import tritonclient.http as httpclient

                self._client = httpclient.InferenceServerClient(url=self._url)
                self._infer_input_cls = httpclient.InferInput
                self._infer_output_cls = httpclient.InferRequestedOutput
            except ImportError:
                raise ImportError(
                    "tritonclient[http] is required. "
                    "Install with: pip install tritonclient[http]"
                )

    def _infer(
        self,
        operation: str,
        vectors: np.ndarray,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Send an inference request to Triton.

        Args:
            operation: Operation name (compress, decompress, search, etc.)
            vectors: float32 input array
            extra_params: Additional parameters serialized as JSON string

        Returns:
            Dict of output name -> numpy array
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Build inputs
        input_vectors = self._infer_input_cls(
            "vectors", list(vectors.shape), "FP32"
        )
        input_vectors.set_data_from_numpy(vectors)

        inputs = [input_vectors]

        # Operation as a string input
        op_arr = np.array([operation], dtype=object)
        input_op = self._infer_input_cls("operation", [1], "BYTES")
        input_op.set_data_from_numpy(op_arr)
        inputs.append(input_op)

        # Extra params
        if extra_params:
            params_arr = np.array([json.dumps(extra_params)], dtype=object)
            input_params = self._infer_input_cls("parameters", [1], "BYTES")
            input_params.set_data_from_numpy(params_arr)
            inputs.append(input_params)

        # Request outputs
        outputs = [
            self._infer_output_cls("output"),
        ]

        result = self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=outputs,
        )

        return {"output": result.as_numpy("output")}

    def compress(
        self,
        vector: np.ndarray,
    ) -> Dict[str, Any]:
        """Compress a vector via the Triton-deployed BitPolar model.

        Args:
            vector: float32 array of shape (dim,) or (n, dim)

        Returns:
            Dict with "output" key containing the Triton response
        """
        return self._infer("compress", vector)

    def decompress(
        self,
        codes: np.ndarray,
    ) -> Dict[str, Any]:
        """Decompress codes via the Triton-deployed model.

        Args:
            codes: Compressed code array

        Returns:
            Dict with decompressed vectors
        """
        return self._infer("decompress", codes)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Search stored codes on the Triton server.

        Args:
            query: float32 query vector of shape (dim,) or (nq, dim)
            top_k: Number of results to return

        Returns:
            Dict with search results from the server
        """
        return self._infer("search", query, extra_params={"top_k": top_k})

    def is_server_ready(self) -> bool:
        """Check if the Triton server is ready.

        Returns:
            True if the server is live and the model is ready
        """
        try:
            return self._client.is_server_ready()
        except Exception:
            return False

    def is_model_ready(self) -> bool:
        """Check if the BitPolar model is loaded and ready.

        Returns:
            True if the model is ready for inference
        """
        try:
            return self._client.is_model_ready(self._model_name)
        except Exception:
            return False

    @property
    def url(self) -> str:
        """Triton server URL."""
        return self._url

    @property
    def model_name(self) -> str:
        """Deployed model name."""
        return self._model_name

    def __repr__(self) -> str:
        return (
            f"BitPolarTritonClient(url='{self._url}', "
            f"model_name='{self._model_name}', protocol='{self._protocol}')"
        )
