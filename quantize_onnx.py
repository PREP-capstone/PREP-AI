"""export_onnx.py로 만든 fp32 ONNX를 동적 int8 양자화한다.

사용법:
    python quantize_onnx.py --in model_fp32.onnx --out model_int8.onnx
"""

from __future__ import annotations

import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", default="model_fp32.onnx")
    parser.add_argument("--out", default="model_int8.onnx")
    args = parser.parse_args()

    quantize_dynamic(args.input, args.out, weight_type=QuantType.QInt8)

    before_mb = os.path.getsize(args.input) / (1024 * 1024)
    after_mb = os.path.getsize(args.out) / (1024 * 1024)
    reduction = (1 - after_mb / before_mb) * 100
    print(f"{args.input} ({before_mb:.1f}MB) -> {args.out} ({after_mb:.1f}MB), {reduction:.1f}% 감소")


if __name__ == "__main__":
    main()
