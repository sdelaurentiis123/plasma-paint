"""Transport feature declaration.

The fixed outboard-midplane cache does not contain a physically complete local
cross-field flux. Density alone is never converted into a transport claim.
"""

from __future__ import annotations


def unavailable_transport() -> dict:
    return {
        "available": False,
        "kind": None,
        "reason": "fixed-plane source lacks a justified local cross-field transport diagnostic",
        "visualization_proxy_used_as_transport": False,
    }

