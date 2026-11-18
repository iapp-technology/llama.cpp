# Nemotron 3.5 Parse (and OpenThai OCR Markdown) in llama.cpp

This branch (`nemotron-parse`) adds NVIDIA **Nemotron 3.5 Parse** (`RadioMBartPrefixForCausalLM`) to llama.cpp: a
C-RADIOv4-SO400M vision encoder with a neck whose output is the causal prefix of a 10-layer mBART-style decoder, which
writes a page as Markdown blocks with their boxes and classes. It runs NVIDIA's model and fine-tunes of it, such as
iApp's OpenThai OCR Markdown (`iapp/OpenThai2.0-OCR-Markdown`, branch `v2.1` for OpenThai2.1).

**Not in upstream llama.cpp.** This code was written with an AI assistant; llama.cpp does not accept AI-written pull
requests, so it is kept as a fork. Ready-made GGUFs: `iapp/OpenThai2.0-OCR-Markdown-GGUF` on Hugging Face.

## Build

```bash
git clone -b nemotron-parse https://github.com/iapp-technology/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release            # Metal on macOS, CPU elsewhere; add -DGGML_CUDA=ON for NVIDIA
cmake --build build -j --target llama-mtmd-cli llama-server llama-quantize
```

If the configure step cannot download the server's web UI assets (an offline or source-only checkout), add
`-DLLAMA_USE_PREBUILT_UI=OFF`.

## Convert a checkpoint

```bash
pip install -r requirements.txt   # or at least: gguf-py (this repo), torch, transformers>=5.14, safetensors
python convert_hf_to_gguf.py MODEL_DIR --outtype f16 --outfile text-f16.gguf
python convert_hf_to_gguf.py MODEL_DIR --mmproj --outtype f16 --outfile mmproj-f16.gguf
build/bin/llama-quantize text-f16.gguf text-q8_0.gguf Q8_0      # the decoder; keep the mmproj at F16
```

`MODEL_DIR` is a Hugging Face checkout of `nvidia/Nemotron-3.5-Parse` or a fine-tune of it. The text GGUF carries:
- a chat template that puts the image first, then `\n`, then the task tokens, which is how vLLM builds the prompt,
  whatever the order of the message parts;
- the sampling of NVIDIA's `DocumentParser` as defaults: temperature 0.2, top-k 50, top-p 1.0, min-p 0, repetition
  penalty 1.05 over 8,760 tokens. The checkpoint's `generation_config.json` asks for 1.1, which is for HF `generate()`;
  over llama.cpp's default 64-token window it turns a page's second bullet or a repeated block class into another token.

## Run

One page (the GGUF's default sampling, as `DocumentParser` asks vLLM for it; add `--temp 0` for greedy):

```bash
build/bin/llama-mtmd-cli -m text-q8_0.gguf --mmproj mmproj-f16.gguf --image page.png --jinja -n 8744 -c 16384 \
  -p $'<__media__>\n</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>'
```

The output is the model's raw form, `<x_..><y_..>text<x_..><y_..><class_..>` per block. Turn it into Markdown with the
`nemotron_parse` package that comes with the model (`nemotron_parse.postprocessing.postprocess_document`).

As a server, for NVIDIA's `DocumentParser` (same prompt, same post-processing as with vLLM):

```bash
build/bin/llama-server -m text-q8_0.gguf --mmproj mmproj-f16.gguf --jinja -c 16384 --port 8000
```

```python
from nemotron_parse.inference import DocumentParser, GenerationSettings
with DocumentParser(base_url="http://localhost:8000/v1", model="nemotron-parse") as parser:
    doc = parser.parse("page.png")          # doc["markdown"], doc["blocks"]
```

- llama-server reads `repeat_penalty`, not vLLM's `repetition_penalty`, so the penalty comes from the GGUF's default
  (1.05 over 8,760 tokens). `--repeat-last-n -1` does not mean "whole context" in this llama.cpp; it disables the
  penalty.
- For deterministic output, pass `fullpage=GenerationSettings(temperature=0.0, repetition_penalty=1.0)` and start the
  server with `--repeat-penalty 1.0`.
- `-c 16384` leaves room for the image prefix (up to about 3,400 tokens) and `DocumentParser`'s 8,744-token limit.

## Precision and parity

- Greedy decodes of the F32 GGUF match the PyTorch F32 reference token for token, on CPU in exact F32
  (`-ngl 0 --no-mmproj-offload -fa off -ctk f32 -ctv f32`) and on Metal with default settings: 5 development pages,
  600 tokens each (or to the end of the page), including pages that are resized. Compare with `--repeat-penalty 1.0`
  and no penalty in the reference.
- On CPU in exact F32 the encoder output matches the reference to about 1e-4. GPU backends compute some steps at half
  precision (flash attention, Metal's matrix kernels), as F16 weights do.
- The position-embedding resampling is kept at full precision on every backend (Metal's matrix kernel rounded the
  128 x 128 grid to half, about 1% error on every patch).
- Parity tools (environment variables; for development only):
  - `MTMD_DEBUG_EMBEDDINGS=<file>` writes the encoder output: int32 n_tokens, int32 n_embd, then float32 data.
  - `MTMD_DEBUG_INPUT=<file>` writes the normalized input image (R, G, B planes).
  - `NPARSE_DEBUG_STAGE=pos|vitin|vitout`, with `MTMD_DEBUG_EMBEDDINGS`, makes the vision graph output that intermediate
    tensor, writes it and exits.

## Where the code is

| Part | Files |
|---|---|
| Converter | `conversion/nemotron_parse.py` (+ map entries in `conversion/__init__.py`), `gguf-py/gguf/constants.py` |
| Decoder | `src/models/nemotron-parse.cpp`, `src/models/models.h`, `src/llama-arch.*`, `src/llama-model.cpp` |
| Tokenizer | `src/llama-vocab.*` (pre-type `nemotron-parse`) |
| Vision graph | `tools/mtmd/models/nemotron-parse.cpp` (graph + `pos_weights`), `tools/mtmd/models/models.h`, `tools/mtmd/mtmd-graph.cpp` |
| Vision loading | `tools/mtmd/mtmd-model.cpp`, `tools/mtmd/mtmd-impl.h` |
| Preprocessing | `tools/mtmd/mtmd-image.{h,cpp}` (`mtmd_image_preprocessor_nemotron_parse`), `tools/mtmd/mtmd.cpp` |
| Graph inputs | `tools/mtmd/mtmd-subcontext.cpp` (`pos_ax`, `pos_ay`; the content size travels in `anyres.orig_nx/ny`) |

Model facts the port depends on (no position encoding in the decoder, embeddings x 32, GQA 16/8, the neck's 1 x 4
merge, the summary token, the preprocessing) are listed in the GGUF workspace README
(`git.iapp.co.th:iapp-ai/openthai2.0-ocr-markdown-gguf`).
