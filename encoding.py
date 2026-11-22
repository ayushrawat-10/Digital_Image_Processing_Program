"""
=============================================================================
Module: encoding.py
Description: Master Encoding Module (Imports from Individual Compression Modules)
Course: Digital Image Processing (DIP)
=============================================================================

This module exposes both lossless compression algorithms:
- compression_huffman.py    -> HuffmanNode, build_huffman_tree, huffman_encode, huffman_decode, demonstrate_huffman
- compression_arithmetic.py -> arithmetic_encode, arithmetic_decode, demonstrate_arithmetic
=============================================================================
"""

from compression_huffman import (
    HuffmanNode,
    calculate_frequencies,
    build_huffman_tree,
    generate_huffman_codes,
    huffman_encode,
    huffman_decode,
    demonstrate_huffman,
    demonstrate_image_huffman
)

from compression_arithmetic import (
    calculate_symbol_intervals,
    arithmetic_encode,
    arithmetic_decode,
    demonstrate_arithmetic
)

__all__ = [
    'HuffmanNode',
    'calculate_frequencies',
    'build_huffman_tree',
    'generate_huffman_codes',
    'huffman_encode',
    'huffman_decode',
    'demonstrate_huffman',
    'demonstrate_image_huffman',
    'calculate_symbol_intervals',
    'arithmetic_encode',
    'arithmetic_decode',
    'demonstrate_arithmetic'
]
