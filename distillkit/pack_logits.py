import os

import click
import datasets


def pack_pass(data: dict[str, list], max_len: int) -> dict[str, list]:
    # pack a set of rows into a (hopefully smaller) set of batches
    # truncate rows that are too long
    row_lens = [
        (len(input_ids), idx) for (idx, input_ids) in enumerate(data["input_ids"])
    ]
    row_lens.sort()
    data_out = {key: [] for key in data}
    data_out["attention_mask"] = []

    while row_lens and row_lens[-1][0] >= max_len:
        row_l, idx = row_lens.pop()
        for key in data:
            if key == "attention_mask":
                continue
            data_out[key].append(data[key][idx][:max_len])
            assert len(data_out[key][-1]) == max_len, (
                f"{key} {len(data_out[key][-1])} != {max_len}"
            )
        data_out["attention_mask"].append([1] * max_len)

    if not row_lens:
        return data_out

    # greedily pack remaining rows
    # take longest first then fill with shortest
    row_lens = [row_lens[-1]] + row_lens[:-1]
    current_batch = {key: [] for key in data}
    current_batch_indices = []
    current_batch["attention_mask"] = []
    current_token_count = 0
    current_num_examples = 0
    while row_lens:
        _, idx = row_lens.pop(0)
        space = max_len - current_token_count
        tokens_to_take = min(space, len(data["input_ids"][idx]))
        for key in data:
            if key == "attention_mask":
                continue
            current_batch[key] += data[key][idx][:space]
        current_batch["attention_mask"] += [current_num_examples + 1] * tokens_to_take
        current_batch_indices.append(idx)
        current_token_count += tokens_to_take
        current_num_examples += 1
        if current_token_count >= max_len:
            token_ct = len(current_batch["input_ids"])
            for key in current_batch:
                assert len(current_batch[key]) == token_ct, (
                    f"{key} {len(current_batch[key])} != {token_ct}"
                )
                data_out[key].append(current_batch[key])
            current_batch = {key: [] for key in data}
            current_batch["attention_mask"] = []
            current_batch_indices = []
            current_token_count = 0
            current_num_examples = 0
            if row_lens:
                row_lens = [row_lens[-1]] + row_lens[:-1]
    if current_num_examples > 0:
        for idx in current_batch_indices:
            for key in data:
                if key == "attention_mask":
                    continue
                data_out[key].append(data[key][idx][:max_len])
            data_out["attention_mask"].append([1] * len(data_out["input_ids"][-1]))
    return data_out


def _truncate_row(row):
    res = {}
    if "token_ids" in row:
        expected_len = len(row["token_ids"])
    elif "packed_indices" in row:
        expected_len = len(row["packed_indices"])
    elif "bytepacked_indices" in row:
        expected_len = len(row["bytepacked_indices"])
    else:
        raise ValueError("No token_ids or packed_indices found in row")
    for key in ["input_ids", "labels", "attention_mask"]:
        if key in row:
            res[key] = row[key][:expected_len]
    return res


def iterative_packing(
    ds: datasets.Dataset,
    max_len: int,
    num_proc: int | None = None,
    max_iters: int = 4,
    batch_size: int = 32,
) -> datasets.Dataset:
    # truncate input_ids to match packed_indices
    # because if the logits came from vllm, we may not have logits for the final token
    ds = ds.map(
        _truncate_row,
        num_proc=num_proc,
        desc="Truncating input_ids, labels, and attention_mask",
    )

    ds_out = []
    ds_current = ds
    print(f"Batch size: {batch_size}")
    for iter_idx in range(max_iters):
        print(f"len(ds_current): {len(ds_current)}")
        print(f"len(ds_out): {len(ds_out)}")
        if num_proc:
            batched_num_proc = min(num_proc, len(ds_current) // batch_size)
        else:
            batched_num_proc = None
        if batched_num_proc is not None and batched_num_proc < 1:
            batched_num_proc = None
        print(f"Starting iteration {iter_idx}")
        print(f"num_proc: {num_proc}, batched_num_proc: {batched_num_proc}")
        ds_p = ds_current.map(
            pack_pass,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={"max_len": max_len},
            num_proc=batched_num_proc,
            desc=f"Packing iteration {iter_idx}",
        )
        ds_done = ds_p.filter(
            lambda x: len(x) == max_len,
            num_proc=num_proc,
            input_columns=["input_ids"],
            desc="Finding full batches",
        )
        ds_current = ds_p.filter(
            lambda x: len(x) != max_len,
            num_proc=num_proc,
            input_columns=["input_ids"],
            desc="Finding partial batches",
        )
        print(
            f"Finished iteration {iter_idx}, {len(ds_done)} batches packed, {len(ds_current)} examples remaining"
        )
        ds_out.append(ds_done)
        if len(ds_current) <= 1:
            break

    if len(ds_current) > 0:
        ds_out.append(ds_current)
    return datasets.concatenate_datasets(ds_out)


@click.command("distillkit-pack-logits")
@click.option("--dataset", type=str, required=True)
@click.option("--split", type=str, required=False)
@click.option("--max-len", type=int, required=True)
@click.option("--output", type=str, required=True)
@click.option("--num-proc", type=int, default=None)
@click.option("--shuffle-seed", type=int, default=None)
@click.option("--max-iters", type=int, default=4)
@click.option("--batch-size", type=int, default=32)
@click.option("--remove-columns", type=str, multiple=True)
def pack_logits_cli(
    dataset: str,
    split: str | None,
    max_len: int,
    output: str,
    num_proc: int | None,
    shuffle_seed: int | None,
    max_iters: int,
    batch_size: int,
    remove_columns: list[str] | None,
):
    if os.path.exists(dataset):
        ds = datasets.load_from_disk(dataset)
        if split is not None:
            assert isinstance(ds, datasets.DatasetDict), (
                "Expected a DatasetDict, got a Dataset"
            )
            ds = ds[split]
    else:
        ds = datasets.load_dataset(dataset, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if remove_columns:
        ds = ds.remove_columns(remove_columns)
    ds_out = iterative_packing(
        ds, max_len, num_proc=num_proc, max_iters=max_iters, batch_size=batch_size
    )
    ds_out.save_to_disk(output)


if __name__ == "__main__":
    pack_logits_cli()
