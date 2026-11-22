"""
=============================================================================
Module: compression_huffman.py
Description: Manual Huffman Lossless Encoding & Decoding from Scratch
Course: Digital Image Processing (DIP)
=============================================================================

Theory:
- Variable-length prefix code based on symbol frequencies.
- High-frequency symbols receive shorter binary codes; rare symbols receive longer codes.
- No code is a prefix of another code (prefix property), allowing unambiguous decoding.

Execution:
    Run standalone:  python compression_huffman.py
=============================================================================
"""

import math


class HuffmanNode:
    """
    A simple node in the Huffman Tree.
    """
    def __init__(self, char, freq, left=None, right=None):
        self.char = char          # Character (None for internal nodes)
        self.freq = freq          # Frequency of the character / subtree
        self.left = left          # Left child (represents bit '0')
        self.right = right        # Right child (represents bit '1')

    def __lt__(self, other):
        return self.freq < other.freq


def calculate_frequencies(text):
    """
    Count the frequency of each unique symbol in the input text.
    """
    freq_dict = {}
    for char in text:
        freq_dict[char] = freq_dict.get(char, 0) + 1
    return freq_dict


def build_huffman_tree(freq_dict):
    """
    Construct the binary Huffman tree from symbol frequencies.
    """
    nodes = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]

    if len(nodes) == 1:
        root = HuffmanNode(None, nodes[0].freq, left=nodes[0])
        return root

    while len(nodes) > 1:
        nodes.sort(key=lambda n: n.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        parent = HuffmanNode(char=None, freq=left.freq + right.freq, left=left, right=right)
        nodes.append(parent)

    return nodes[0]


def generate_huffman_codes(root):
    """
    Traverse the Huffman tree recursively to assign binary codes.
    """
    codes = {}

    def _traverse(node, current_code):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code if current_code != "" else "0"
            return
        _traverse(node.left, current_code + "0")
        _traverse(node.right, current_code + "1")

    _traverse(root, "")
    return codes


def huffman_encode(text):
    """
    Encode input string using Huffman coding.
    """
    if not text:
        return "", None, {}, {}

    freq_dict = calculate_frequencies(text)
    root = build_huffman_tree(freq_dict)
    codes = generate_huffman_codes(root)

    encoded_bits = "".join(codes[char] for char in text)

    total_chars = len(text)
    original_bits = total_chars * 8
    compressed_bits = len(encoded_bits)
    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 1.0
    space_saved = ((original_bits - compressed_bits) / original_bits) * 100.0

    entropy = 0.0
    avg_length = 0.0
    for char, freq in freq_dict.items():
        prob = freq / total_chars
        entropy += -prob * math.log2(prob)
        avg_length += prob * len(codes[char])

    efficiency = (entropy / avg_length * 100.0) if avg_length > 0 else 100.0

    stats = {
        "total_chars": total_chars,
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "compression_ratio": compression_ratio,
        "space_saved_percent": space_saved,
        "entropy": entropy,
        "avg_code_length": avg_length,
        "efficiency_percent": efficiency,
        "frequencies": freq_dict
    }

    return encoded_bits, root, codes, stats


def huffman_decode(encoded_bits, root):
    """
    Decode a binary bit sequence back to text using the Huffman tree.
    """
    if not encoded_bits or root is None:
        return ""

    decoded_chars = []
    current_node = root

    for bit in encoded_bits:
        if bit == '0':
            current_node = current_node.left
        elif bit == '1':
            current_node = current_node.right

        if current_node.char is not None:
            decoded_chars.append(current_node.char)
            current_node = root

    if all(isinstance(c, str) for c in decoded_chars):
        return "".join(decoded_chars)
    return decoded_chars


def demonstrate_huffman(sample_text="BANANA"):
    """
    Run and print an educational demonstration of Huffman coding.
    """
    print("\n" + "=" * 65)
    print("           DEMONSTRATION: HUFFMAN ENCODING & DECODING           ")
    print("=" * 65)
    print(f"Original Input Text : '{sample_text}'")

    encoded_bits, root, codes, stats = huffman_encode(sample_text)
    decoded_text = huffman_decode(encoded_bits, root)

    print("\n1. Symbol Frequencies & Generated Huffman Codes:")
    print("-" * 55)
    print(f"{'Symbol':^10} | {'Frequency':^12} | {'Probability':^12} | {'Huffman Code':^14}")
    print("-" * 55)
    for char in sorted(stats['frequencies'].keys()):
        freq = stats['frequencies'][char]
        prob = freq / stats['total_chars']
        code = codes[char]
        print(f"{repr(char):^10} | {freq:^12} | {prob:^12.4f} | {code:^14}")
    print("-" * 55)

    print(f"\n2. Encoded Bit Sequence : {encoded_bits}")
    print(f"3. Decoded Output Text  : '{decoded_text}'")
    print(f"4. Verification Match   : {'SUCCESS (Matches Original)' if decoded_text == sample_text else 'FAILED'}")

    print("\n5. Compression Performance Metrics:")
    print(f"   - Original Size (ASCII 8-bit) : {stats['original_bits']} bits")
    print(f"   - Compressed Bit Length       : {stats['compressed_bits']} bits")
    print(f"   - Compression Ratio           : {stats['compression_ratio']:.2f} : 1")
    print(f"   - Space Saved                 : {stats['space_saved_percent']:.2f}%")
    print(f"   - Information Entropy (H)     : {stats['entropy']:.4f} bits/symbol")
    print(f"   - Average Code Length (L_avg) : {stats['avg_code_length']:.4f} bits/symbol")
    print(f"   - Coding Efficiency           : {stats['efficiency_percent']:.2f}%")
    print("=" * 65)
    return stats


def demonstrate_image_huffman(image_patch=None):
    """
    Demonstrate Huffman compression applied directly to image pixel intensities!
    In DIP:
      - The symbols are pixel gray levels (0 to 255).
      - The symbol frequencies are the Image Histogram.
      - The probabilities are the Normalized Histogram.
    """
    import numpy as np
    if image_patch is None:
        # Create a small representative 8x8 image patch
        image_patch = np.array([
            [120, 120, 120, 120, 150, 150, 200, 200],
            [120, 120, 120, 150, 150, 150, 200, 200],
            [120, 120, 150, 150, 150, 200, 200, 200],
            [120, 150, 150, 150, 200, 200, 200, 200],
            [150, 150, 150, 200, 200, 200, 220, 220],
            [150, 150, 200, 200, 200, 220, 220, 220],
            [150, 200, 200, 200, 220, 220, 220, 220],
            [200, 200, 200, 220, 220, 220, 220, 220],
        ], dtype=np.uint8)

    pixels = image_patch.flatten().tolist()
    print("\n" + "=" * 65)
    print("      DEMONSTRATION: HUFFMAN ENCODING ON IMAGE PIXEL DATA      ")
    print("=" * 65)
    print(f"Image Patch Dimensions: {image_patch.shape} ({len(pixels)} total pixels)")

    encoded_bits, root, codes, stats = huffman_encode(pixels)
    decoded_pixels = huffman_decode(encoded_bits, root)

    print("\n1. Pixel Histogram (Frequencies) & Generated Variable-Length Codes:")
    print("-" * 65)
    print(f"{'Pixel Value':^12} | {'Count (Hist)':^14} | {'Probability':^14} | {'Huffman Code':^16}")
    print("-" * 65)
    # Sort by frequency descending (most frequent pixel first)
    sorted_by_freq = sorted(stats['frequencies'].items(), key=lambda x: x[1], reverse=True)
    for val, freq in sorted_by_freq:
        prob = freq / stats['total_chars']
        code = codes[val]
        print(f"{val:^12} | {freq:^14} | {prob:^14.4f} | {code:^16}")
    print("-" * 65)

    print("\n2. Encoded Bitstream Sample (first 50 bits):")
    print(f"   {encoded_bits[:50]}...")
    print(f"3. Decoded Pixel Verification : {'SUCCESS (100% Exact Match)' if decoded_pixels == pixels else 'FAILED'}")

    print("\n4. Compression Performance Metrics on Image Data:")
    print(f"   - Original Uncompressed Size  : {stats['original_bits']} bits (8 bits/pixel)")
    print(f"   - Huffman Compressed Size     : {stats['compressed_bits']} bits")
    print(f"   - Average Bits Per Pixel (bpp): {stats['avg_code_length']:.2f} bits/pixel (vs 8.00 bpp)")
    print(f"   - Compression Ratio           : {stats['compression_ratio']:.2f} : 1")
    print(f"   - Storage Space Saved         : {stats['space_saved_percent']:.2f}%")
    print(f"   - Theoretical Entropy (H)     : {stats['entropy']:.4f} bits/pixel")
    print(f"   - Coding Efficiency           : {stats['efficiency_percent']:.2f}%")
    print("=" * 65)
    return stats


if __name__ == "__main__":
    demonstrate_huffman("BANANA")
    demonstrate_image_huffman()
