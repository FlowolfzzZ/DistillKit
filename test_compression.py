import json

import datasets
import torch
import tqdm
import transformers
from pydantic import BaseModel
from transformers.modeling_outputs import CausalLMOutput

from distillkit.compression import (
    DistributionQuantizationConfig,
    LegacyLogitCompressionConfig,
    LogprobCompressor,
    QuantizationBin,
    densify,
)
from distillkit.compression.config import TermDtype
from distillkit.configuration import MissingProbabilityHandling


class ErrorStats(BaseModel):
    mean: float
    std: float
    max: float
    min: float

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "ErrorStats":
        return cls(
            mean=tensor.mean().item(),
            std=tensor.std().item(),
            max=tensor.max().item(),
            min=tensor.min().item(),
        )

    def __str__(self):
        return (
            f"mean: {self.mean:.4e}, "
            f"std: {self.std:.4e}, "
            f"max: {self.max:.4e}, "
            f"min: {self.min:.4e}"
        )


class CompressionEvalResult(BaseModel):
    bytes_per_token: float
    compression_ratio: float
    kld_fwd: ErrorStats
    kld_bwd: ErrorStats
    jsd: ErrorStats
    prob_mse: ErrorStats

    def __str__(self):
        return (
            f"Bytes per token: {self.bytes_per_token:.4f}\n"
            f"Compression Ratio: {self.compression_ratio:.4f}\n"
            f"KLD Forward: {self.kld_fwd}\n"
            f"KLD Backward: {self.kld_bwd}\n"
            f"JSD: {self.jsd}\n"
            f"Prob MSE: {self.prob_mse}\n"
        )


def generate_test_logprobs(
    model: str,
    max_seq_len: int,
    samples: int,
    batch_size: int = 256,
    dataset: str = "TEL-LLM/fineweb-edu-1M",
    device: str = "cuda",
) -> list[dict[str, torch.Tensor]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        device_map=device,
    )
    if max_seq_len < 1 or max_seq_len > model.config.max_position_embeddings:
        print(
            f"Clamping max_seq_len from {max_seq_len} to {model.config.max_position_embeddings}"
        )
        max_seq_len = model.config.max_position_embeddings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = (
        datasets.load_dataset(
            dataset,
            split="train",
        )
        .shuffle(seed=42)
        .select(range(samples))
    )
    ds = ds.map(
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=max_seq_len, padding="max_length"
        ),
        batched=True,
        remove_columns=["text"],
    )
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def _generate_rows():
        with torch.no_grad():
            for i_0 in tqdm.tqdm(
                range(0, len(ds), batch_size), desc="Generating logprobs"
            ):
                i_1 = min(i_0 + batch_size, len(ds))
                batch = ds[i_0:i_1]
                outputs: CausalLMOutput = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    return_dict=True,
                )
                logits = outputs.logits
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()
                del logits, outputs
                for i in range(len(batch["input_ids"])):
                    yield {
                        "attention_mask": batch["attention_mask"][i, :].cpu(),
                        "logprobs": logprobs[i, :, :],
                    }

    return list(_generate_rows())


def jensen_shannon_divergence(
    log_p: torch.Tensor, log_q: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    """
    p = torch.nn.functional.softmax(log_p, dim=dim)
    del log_p
    q = torch.nn.functional.softmax(log_q, dim=dim)
    del log_q
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-9).log()
    del m
    jsd = 0.5 * (
        torch.nn.functional.kl_div(log_m, p, log_target=False, reduction="none")
        + torch.nn.functional.kl_div(log_m, q, log_target=False, reduction="none")
    )
    return jsd


def eval_compression_quality(
    logprob_rows: list[dict[str, torch.Tensor]],
    config: DistributionQuantizationConfig | LegacyLogitCompressionConfig,
    batch_size: int = 256,
    device: str = "cuda",
    missing_prob_handling: MissingProbabilityHandling = MissingProbabilityHandling.ZERO,
) -> CompressionEvalResult:
    compressor = (
        LogprobCompressor(
            config=config
            if isinstance(config, DistributionQuantizationConfig)
            else None,
            legacy_config=(
                config if isinstance(config, LegacyLogitCompressionConfig) else None
            ),
        )
        if config != "bf16"
        else None
    )

    vocab_size = None
    bytes_per_token = None
    if config == "bf16":
        vocab_size = VOCAB_SIZE
        bytes_per_token = 2 * vocab_size
    elif isinstance(config, DistributionQuantizationConfig):
        vocab_size = config.d
        bytes_per_token = (config.total_bits() + 7) // 8
    elif isinstance(config, LegacyLogitCompressionConfig):
        vocab_size = config.vocab_size
        bytes_per_token = compressor.legacy_compressor.bytes_per_token()
    else:
        raise ValueError("Invalid config type")

    kld_fwd_list = []
    kld_bwd_list = []
    jsd_list = []
    mse_list = []
    for i_0 in tqdm.tqdm(range(0, len(logprob_rows), batch_size), desc="Evaluating"):
        i_1 = min(i_0 + batch_size, len(logprob_rows))
        batch = logprob_rows[i_0:i_1]
        logprobs = torch.stack([row["logprobs"] for row in batch], dim=0).to(device)
        mask = (torch.stack([row["attention_mask"] for row in batch], dim=0) > 0).to(
            device
        )
        logprobs = logprobs.float().masked_fill(mask.unsqueeze(-1), -1e6)

        if config != "bf16":
            batch_out = compressor.compress(logprobs)
            rt_ids, rt_logp = compressor.decompress_to_sparse(batch_out)
            del batch_out
            rt_logp_dense = densify(
                rt_ids,
                rt_logp.float(),
                vocab_size,
                missing=missing_prob_handling,
                renormalize=True,
                fill_value=-1e12,
            )
            del rt_ids, rt_logp
        else:
            rt_logp_dense = logprobs.to(torch.bfloat16).float()
        rt_logp_dense = rt_logp_dense.masked_fill(mask.unsqueeze(-1), -1e6)

        kld_fwd = torch.nn.functional.kl_div(
            logprobs,
            rt_logp_dense,
            log_target=True,
            reduction="none",
        ).masked_fill(mask.unsqueeze(-1), 0.0)
        kld_fwd = kld_fwd.sum(dim=-1)
        kld_bwd = torch.nn.functional.kl_div(
            rt_logp_dense,
            logprobs,
            log_target=True,
            reduction="none",
        ).masked_fill(mask.unsqueeze(-1), 0.0)
        kld_bwd = kld_bwd.sum(dim=-1)

        jsd = jensen_shannon_divergence(logprobs, rt_logp_dense).sum(dim=-1)
        probs = torch.exp(logprobs)
        rt_probs = torch.exp(rt_logp_dense)
        mse = (probs - rt_probs) ** 2
        mse = mse.mean(dim=-1)

        kld_fwd_list.append(kld_fwd.cpu())
        kld_bwd_list.append(kld_bwd.cpu())
        jsd_list.append(jsd.cpu())
        mse_list.append(mse.cpu())

    compression_ratio = bytes_per_token / (vocab_size * 4)
    return CompressionEvalResult(
        bytes_per_token=bytes_per_token,
        compression_ratio=compression_ratio,
        kld_fwd=ErrorStats.from_tensor(torch.cat(kld_fwd_list)),
        kld_bwd=ErrorStats.from_tensor(torch.cat(kld_bwd_list)),
        jsd=ErrorStats.from_tensor(torch.cat(jsd_list)),
        prob_mse=ErrorStats.from_tensor(torch.cat(mse_list)),
    )


MODEL_NAME = "Qwen/Qwen3-8B-Base"  # "gpt2-xl"
VOCAB_SIZE = 151936  # 50257
MAX_SEQ_LEN = 256

TEST_CONFIGS: list[DistributionQuantizationConfig | LegacyLogitCompressionConfig] = [
    "bf16",
    *[
        DistributionQuantizationConfig(
            d=VOCAB_SIZE,
            k=num,
            exact_k=num,
            exact_dtype="bfloat16",
            polynomial_terms=[],
            residual_bins=[],
            term_dtype="float32",
            delta_encoding=False,
            error_diffusion=False,
        )
        for num in [1, 2, 4, 32, 50, 64, 128]
    ],
    LegacyLogitCompressionConfig(
        k=4096,
        exact_k=4096,
        polynomial_degree=0,
        vocab_size=VOCAB_SIZE,
    ),
    LegacyLogitCompressionConfig(
        k=128,
        exact_k=32,
        polynomial_degree=8,
        invert_polynomial=True,
        with_sqrt_term=False,
        vocab_size=VOCAB_SIZE,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=4096,
        exact_k=16,
        exact_dtype="bfloat16",
        polynomial_terms=[0, 1, 2, 3, 4, "sqrt"],
        term_dtype="float32",
        residual_bins=[],
        delta_encoding=False,
        error_diffusion=False,
        normalize_t=True,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=4096,
        exact_k=16,
        exact_dtype="bfloat16",
        polynomial_terms=[0, 1, 2, "sqrt"],
        term_dtype="float32",
        residual_bins=[
            QuantizationBin(
                scale_dtype=TermDtype.FLOAT16,
                element_bits=8,
                num_elements=16,
            ),
            QuantizationBin(
                scale_dtype=TermDtype.FLOAT16,
                element_bits=2,
                num_elements=496,
            ),
        ],
        delta_encoding=False,
        error_diffusion=False,
        normalize_t=True,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=4096,
        exact_k=16,
        exact_dtype="bfloat16",
        polynomial_terms=[0, 1, 2, "sqrt"],
        term_dtype="float32",
        residual_bins=[
            QuantizationBin(
                scale_dtype=TermDtype.FLOAT16,
                element_bits=8,
                num_elements=16,
            ),
            QuantizationBin(
                scale_dtype=TermDtype.FLOAT16,
                element_bits=2,
                num_elements=496,
            ),
        ],
        delta_encoding=False,
        error_diffusion=True,
        normalize_t=True,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=50,
        exact_k=1,
        polynomial_terms=[0, 1, "sqrt"],
        term_dtype=TermDtype.FLOAT32,
        exact_dtype=TermDtype.BFLOAT16,
        residual_bins=[],
        delta_encoding=True,
        error_diffusion=False,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=50,
        exact_k=1,
        polynomial_terms=[0, 1, "sqrt"],
        term_dtype=TermDtype.FLOAT32,
        exact_dtype=TermDtype.BFLOAT16,
        residual_bins=[],
        delta_encoding=False,
        error_diffusion=False,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=128,
        exact_k=24,
        polynomial_terms=[0, 1, 2, 3, 4, "sqrt"],
        term_dtype=TermDtype.FLOAT32,
        exact_dtype=TermDtype.BFLOAT16,
        residual_bins=[],
        delta_encoding=False,
        error_diffusion=False,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=128,
        exact_k=24,
        polynomial_terms=[0, 1, 2, 3, 4, "sqrt"],
        term_dtype=TermDtype.FLOAT32,
        exact_dtype=TermDtype.BFLOAT16,
        residual_bins=[],
        delta_encoding=False,
        error_diffusion=False,
        normalize_t=True,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=128,
        exact_k=16,
        polynomial_terms=[0, 1, 2, 3, 4, "sqrt"],
        term_dtype=TermDtype.FLOAT32,
        exact_dtype=TermDtype.BFLOAT16,
        residual_bins=[],
        delta_encoding=True,
        error_diffusion=False,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=128,
        exact_k=16,
        polynomial_terms=[0, 1, 2, 3, 4, "sqrt"],
        term_dtype=TermDtype.FLOAT32,
        exact_dtype=TermDtype.BFLOAT16,
        residual_bins=[],
        delta_encoding=False,
        error_diffusion=False,
    ),
    DistributionQuantizationConfig(
        d=VOCAB_SIZE,
        k=128,
        exact_k=16,
        polynomial_terms=[0, 1, 2, 3, 4, "sqrt"],
        term_dtype=TermDtype.FLOAT32,
        exact_dtype=TermDtype.BFLOAT16,
        residual_bins=[],
        delta_encoding=False,
        error_diffusion=False,
        normalize_t=True,
    ),
]

if __name__ == "__main__":
    report = []
    print(f"Model: {MODEL_NAME}")
    logprob_rows = generate_test_logprobs(
        MODEL_NAME,
        MAX_SEQ_LEN,
        1024,
        batch_size=32,
        dataset="TEL-LLM/fineweb-edu-1M",
        device="cuda:1",
    )
    for mph in [
        MissingProbabilityHandling.ZERO,
        # MissingProbabilityHandling.SYMMETRIC_UNIFORM,
    ]:
        for config in TEST_CONFIGS:
            if isinstance(config, dict):
                config = DistributionQuantizationConfig.model_validate(config)
            print(f"Testing config: {repr(config)}, mph: {mph}")
            result = eval_compression_quality(
                logprob_rows,
                config,
                batch_size=32,
                device="cuda:4",
                missing_prob_handling=mph,
            )
            print(result)
            print("-" * 80)
            print()

            if config == "bf16":
                report.append(
                    {
                        "kind": "bf16",
                        "config": {},
                        "missing_prob_handling": mph.value,
                        "result": result.model_dump(mode="json"),
                    }
                )
            elif config.k == config.exact_k:
                report.append(
                    {
                        "kind": "top_k",
                        "config": {"k": config.k},
                        "missing_prob_handling": mph.value,
                        "result": result.model_dump(mode="json"),
                    }
                )
            else:
                report.append(
                    {
                        "kind": (
                            "legacy"
                            if isinstance(config, LegacyLogitCompressionConfig)
                            else "distrib_quant"
                        ),
                        "config": config.model_dump(mode="json"),
                        "missing_prob_handling": mph.value,
                        "result": result.model_dump(mode="json"),
                    }
                )
    print(json.dumps(report))
