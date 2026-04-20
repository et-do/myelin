"""Myelin — neuromorphic long-term AI memory system."""

from importlib.metadata import version

__version__ = version("myelin-mcp")

# Ensure __version__ is exported.
__all__ = ["__version__"]
