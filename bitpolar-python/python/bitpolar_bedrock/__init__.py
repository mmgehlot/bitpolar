"""BitPolar AWS Bedrock Integration — compressed embeddings via Bedrock runtime.

Wraps the boto3 bedrock-runtime client to automatically compress embedding
responses from Amazon Titan and other Bedrock embedding models.

Usage:
    >>> from bitpolar_bedrock import BitPolarBedrockClient
    >>> client = BitPolarBedrockClient(bits=4)
    >>> result = client.embed_and_compress("Hello world")
"""

from bitpolar_bedrock.middleware import BitPolarBedrockClient

__all__ = ["BitPolarBedrockClient"]
__version__ = "0.3.3"
