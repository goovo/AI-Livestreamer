"""一次性把 ModelScope 的 torch 权重转成 MLX fp16 safetensors（.venv-muse 运行）。

用法: .venv-muse/bin/python tools/convert_musetalk.py
输入: .cache/models/musetalk-torch/{musetalkV15/unet.pth, sd-vae-ft-mse/}
输出: .cache/models/musetalk-mlx-fp16/{unet,vae}.safetensors + config.json
whisper_encoder.safetensors 直接从 hf-mirror 下载 MLX 版，无需转换。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map

ROOT = Path(__file__).resolve().parent.parent
TORCH = ROOT / ".cache" / "models" / "musetalk-torch"
OUT = ROOT / ".cache" / "models" / "musetalk-mlx-fp16"


def main() -> None:
    from fusion_mlx.video.musetalk_mlx.models.unet import UNet2DConditionModel
    from fusion_mlx.video.musetalk_mlx.models.vae import AutoencoderKL
    from fusion_mlx.video.musetalk_mlx.utils.weights import (
        load_unet_weights, load_vae_weights, save_native,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    unet = UNet2DConditionModel()
    load_unet_weights(unet, TORCH / "musetalkV15" / "unet.pth")
    unet.update(tree_map(lambda p: p.astype(mx.float16), unet.parameters()))
    mx.eval(unet.parameters())
    save_native(unet, OUT / "unet.safetensors")
    del unet
    mx.clear_cache()
    print(f"unet 转换完成 {time.time()-started:.0f}s", flush=True)

    vae = AutoencoderKL()
    load_vae_weights(vae, TORCH / "sd-vae-ft-mse")
    vae.update(tree_map(lambda p: p.astype(mx.float16), vae.parameters()))
    mx.eval(vae.parameters())
    save_native(vae, OUT / "vae.safetensors")
    del vae
    mx.clear_cache()
    (OUT / "config.json").write_text(json.dumps({
        "model": "MuseTalk-1.5", "framework": "mlx", "dtype": "float16",
        "scaling_factor": 0.18215, "cross_attention_dim": 384, "quantization": None,
    }, indent=2))
    print(f"全部完成 {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
