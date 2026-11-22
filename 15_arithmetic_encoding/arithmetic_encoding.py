"""
=============================================================================
Module: compression_arithmetic.py
Description: Manual Arithmetic Lossless Encoding & Decoding from Scratch
Course: Digital Image Processing (DIP)
=============================================================================

Theory:
- Encodes an entire sequence into a single fractional number in [0.0, 1.0).
- At each symbol, the current interval [low, high) is narrowed proportionally
  to the symbol's cumulative probability sub-interval.
- Uses Python's decimal.Decimal with 60-digit precision to prevent floating
  point underflow on longer symbol streams.

Execution:
    Run standalone:  python compression_arithmetic.py
=============================================================================
"""

from decimal import Decimal, getcontext
getcontext().prec = 60


def calculate_symbol_intervals(text):
    """
    Calculate probability and cumulative intervals [low, high) for each unique symbol.
    """
    freq_dict = {}
    for char in text:
        freq_dict[char] = freq_dict.get(char, 0) + 1

    total_chars = len(text)
    sorted_chars = sorted(freq_dict.keys())

    intervals = {}
    current_low = Decimal('0.0')

    for char in sorted_chars:
        prob = Decimal(str(freq_dict[char])) / Decimal(str(total_chars))
        current_high = current_low + prob
        intervals[char] = {
            'prob': prob,
            'low': current_low,
            'high': current_high
        }
        current_low = current_high

    return intervals


def arithmetic_encode(text):
    """
    Encode an input string using Arithmetic Coding.
    """
    if not text:
        return Decimal('0.0'), {}, [], Decimal('0.0'), Decimal('1.0')

    intervals = calculate_symbol_intervals(text)

    low = Decimal('0.0')
    high = Decimal('1.0')
    steps = []

    for idx, char in enumerate(text):
        symbol_info = intervals[char]
        sym_low = symbol_info['low']
        sym_high = symbol_info['high']
        range_val = high - low

        new_high = low + range_val * sym_high
        new_low = low + range_val * sym_low

        steps.append({
            'step': idx + 1,
            'char': char,
            'range': range_val,
            'low': new_low,
            'high': new_high
        })

        low = new_low
        high = new_high

    # Midpoint of the final interval
    encoded_value = (low + high) / Decimal('2.0')

    return encoded_value, intervals, steps, low, high


def arithmetic_decode(encoded_value, intervals, text_length):
    """
    Decode an Arithmetic encoded value back to the original string.
    """
    decoded_chars = []
    current_value = Decimal(str(encoded_value))

    for _ in range(text_length):
        found_char = None
        for char, info in intervals.items():
            sym_low = info['low']
            sym_high = info['high']
            if sym_low <= current_value < sym_high:
                found_char = char
                current_value = (current_value - sym_low) / (sym_high - sym_low)
                break

        if found_char is None:
            for char, info in intervals.items():
                if current_value >= info['low']:
                    found_char = char

        decoded_chars.append(found_char)

    if all(isinstance(c, str) for c in decoded_chars):
        return "".join(decoded_chars)
    return decoded_chars


def demonstrate_arithmetic(sample_text="BANANA"):
    """
    Run and print an educational demonstration of Arithmetic coding.
    """
    print("\n" + "=" * 70)
    print("         DEMONSTRATION: ARITHMETIC ENCODING & DECODING         ")
    print("=" * 70)
    print(f"Original Input Text : '{sample_text}' (Length = {len(sample_text)})")

    encoded_val, intervals, steps, final_low, final_high = arithmetic_encode(sample_text)
    decoded_text = arithmetic_decode(encoded_val, intervals, len(sample_text))

    print("\n1. Symbol Cumulative Probability Intervals:")
    print("-" * 65)
    print(f"{'Symbol':^10} | {'Probability':^14} | {'Sub-interval [Low, High)':^35}")
    print("-" * 65)
    for char in sorted(intervals.keys()):
        p = float(intervals[char]['prob'])
        l = float(intervals[char]['low'])
        h = float(intervals[char]['high'])
        print(f"{repr(char):^10} | {p:^14.4f} | [{l:.4f}, {h:.4f})")
    print("-" * 65)

    print("\n2. Step-by-Step Encoding Progress (Interval Narrowing):")
    print("-" * 75)
    print(f"{'Step':^6} | {'Char':^6} | {'Current Low':^28} | {'Current High':^28}")
    print("-" * 75)
    for s in steps:
        print(f"{s['step']:^6} | {repr(s['char']):^6} | {str(s['low'])[:26]:<28} | {str(s['high'])[:26]:<28}")
    print("-" * 75)

    print(f"\n3. Final Encoding Interval : [{final_low}, {final_high})")
    print(f"4. Encoded Floating Tag    : {encoded_val}")
    print(f"5. Decoded Output Text     : '{decoded_text}'")
    print(f"6. Verification Match      : {'SUCCESS (Matches Original)' if decoded_text == sample_text else 'FAILED'}")
    print("=" * 70)
    return encoded_val, decoded_text


if __name__ == "__main__":
    demonstrate_arithmetic("BANANA")
