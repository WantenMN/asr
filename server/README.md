## Models Tree

```
models/
├── faster-whisper-large-v3-turbo-ct2
├── iic
│   ├── punc_ct-transformer_cn-en-common-vocab471067-large
│   ├── speech_fsmn_vad_zh-cn-16k-common-pytorch
│   ├── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
│   └── speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
└── whisper-large-v3-turbo
```

## Run

The following are the available model runners.

```
# Recommended: Fastest
uv run src/paraformer.py

# Paraformer ONNX version
uv run src/paraformer-onnx.py

# Faster-Whisper implementation
uv run src/faster-whisper.py

# Whisper official implementation
uv run src/whisper.py
```
