from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger


# Nemotron 3.5 Parse: C-RADIOv4-SO400M encoder + neck, its output is a causal prefix of a 10-layer mBART-style decoder (no cross-attention)
@ModelBase.register("RadioMBartPrefixForCausalLM")
@ModelBase.example("iapp/OpenThai2.0-OCR-Markdown")
class NemotronParseModel(TextModel):
    model_arch = gguf.MODEL_ARCH.NEMOTRON_PARSE

    _name_map = {
        "self_attn_layer_norm": gguf.MODEL_TENSOR.ATTN_NORM,
        "self_attn.q_proj":     gguf.MODEL_TENSOR.ATTN_Q,
        "self_attn.k_proj":     gguf.MODEL_TENSOR.ATTN_K,
        "self_attn.v_proj":     gguf.MODEL_TENSOR.ATTN_V,
        "self_attn.out_proj":   gguf.MODEL_TENSOR.ATTN_OUT,
        "final_layer_norm":     gguf.MODEL_TENSOR.FFN_NORM,
        "fc1":                  gguf.MODEL_TENSOR.FFN_UP,
        "fc2":                  gguf.MODEL_TENSOR.FFN_DOWN,
    }

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=False)
        vocab_size = self.hparams["vocab_size"]
        reverse_vocab = {id_: tok for tok, id_ in tokenizer.vocab.items()}
        added = tokenizer.added_tokens_decoder

        # normalized added tokens print in their NFKC form in HF (e.g. U+0E33 -> U+0E4D U+0E32), keep that unless the text is taken
        rendered = {i: tokenizer.decode([i]) for i, a in added.items() if not a.special and a.normalized}
        taken = set(reverse_vocab.values())
        n_kept = 0
        for i, text in rendered.items():
            if text != reverse_vocab[i] and text in taken:
                rendered[i] = reverse_vocab[i]
                n_kept += 1
            taken.add(rendered[i])
        if n_kept:
            logger.info(f"{n_kept} normalized added tokens keep their raw text, their NFKC form is another token")

        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            elif i in added:
                tokens.append(rendered.get(i, reverse_vocab[i]))
                # box, class and chart tokens are part of the output text, so only the special ones are CONTROL
                toktypes.append(gguf.TokenType.CONTROL if added[i].special else gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre("nemotron-parse")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # the task prompt brings its own </s><s>
        special_vocab.add_special_token = {"bos": False, "eos": False}
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_eps(1e-5)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        # extra_heads / extra_proj are the MTP draft head
        if not name.startswith("decoder.") or name.startswith(("decoder.extra_heads.", "decoder.extra_proj.")):
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        base, _, suffix = name.rpartition(".")
        suffix = "." + suffix
        if base == "decoder.embed_tokens":
            key = gguf.MODEL_TENSOR.TOKEN_EMBD
        elif base == "decoder.layernorm_embedding":
            key = gguf.MODEL_TENSOR.TOKEN_EMBD_NORM
        elif base == "decoder.layer_norm":
            key = gguf.MODEL_TENSOR.OUTPUT_NORM
        else:
            assert bid is not None, name
            key = self._name_map[base.split(f"decoder.layers.{bid}.", 1)[1]]
        yield self.format_tensor_name(key, bid, suffix), data_torch


@ModelBase.register("RadioMBartPrefixForCausalLM")
class NemotronParseVisionModel(MmprojModel):
    _radio = "encoder.model_encoder.radio_model.model."
    _neck_map = {
        "encoder.conv1":       0,
        "encoder.layer_norm1": 1,
        "encoder.conv2":       2,
        "encoder.layer_norm2": 3,
        "encoder.sum_proj":    4,
        "encoder.layer_norm3": 5,
    }
    _block_map = {
        "norm1":     gguf.MODEL_TENSOR.V_ENC_INPUT_NORM,
        "attn.qkv":  gguf.MODEL_TENSOR.V_ENC_ATTN_QKV,
        "attn.proj": gguf.MODEL_TENSOR.V_ENC_ATTN_O,
        "norm2":     gguf.MODEL_TENSOR.V_ENC_POST_ATTN_NORM,
        "mlp.fc1":   gguf.MODEL_TENSOR.V_ENC_FFN_UP,
        "mlp.fc2":   gguf.MODEL_TENSOR.V_ENC_FFN_DOWN,
    }

    def get_vision_config(self) -> dict | None:
        enc = self.global_config["encoder"]
        if enc["args"]["model"] != "vit_so400m_patch16_224":
            raise ValueError(f"unsupported RADIO backbone {enc['args']['model']!r}")
        return {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "patch_size": enc["patch_size"],
            # longest canvas side; the processor fits pages into 1664 x 2048
            "image_size": self.global_config["virtual_canvas_height"],
        }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.NEMOTRON_PARSE)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if not name.startswith("encoder.") or "input_conditioner" in name or name.endswith("summary_idxs"):
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        base, _, suffix = name.rpartition(".")
        suffix = "." + suffix

        if base in self._neck_map:
            if base == "encoder.conv1":
                data_torch = data_torch.squeeze(-1)
            elif base == "encoder.conv2":
                # (1, 4) conv with stride 4 == linear over 4 neighbour tokens of a row, input index k*1024 + c
                n_out, n_in, _, k = data_torch.shape
                data_torch = data_torch.squeeze(2).permute(0, 2, 1).reshape(n_out, k * n_in)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, self._neck_map[base], suffix), data_torch
            return

        assert name.startswith(self._radio), name
        short = name[len(self._radio):]
        if short == "patch_generator.embedder.weight":
            # linear over flattened (c, y, x) patches -> conv2d kernel
            patch_size = self.hparams["patch_size"]
            data_torch = data_torch.reshape(data_torch.shape[0], 3, patch_size, patch_size)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH), data_torch
        elif short == "patch_generator.pos_embed":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.V_ENC_EMBD_POS), data_torch.squeeze(0)
        elif short == "patch_generator.cls_token.token":
            # 10 tokens: 2 summary (CLS) tokens + 8 registers
            yield self.format_tensor_name(gguf.MODEL_TENSOR.V_ENC_EMBD_CLS, suffix=""), data_torch
        else:
            assert bid is not None, name
            part = short.split(f"blocks.{bid}.", 1)[1].rpartition(".")[0]
            yield self.format_tensor_name(self._block_map[part], bid, suffix), data_torch
