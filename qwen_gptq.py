import argparse
import importlib
import logging
import os
import time
from pprint import pprint
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from datautils import get_loaders
from gptq import GPTQ
from modelutils import find_layers
from quant.QLinear import BaseQuantizeConfig, QLinear
from utils.pack import save_quantized


LOGGER = logging.getLogger(__name__)
SUPPORTED_MODEL_TYPES = {"qwen2_moe", "qwen3_moe"}


def disable_torch_init():
    def skip(*args, **kwargs):
        return None

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def parse_args():
    parser = argparse.ArgumentParser(description="GPTQ quantization for Qwen2-MoE and Qwen3-MoE.")
    parser.add_argument("model", type=str, help="Hugging Face model path or local checkpoint path.")
    parser.add_argument("--attn_bits", "--attn-bits", type=int, default=4, choices=range(1, 9), help="Bit-width used for attention projections.")
    parser.add_argument("--moe_bits", "--moe-bits", type=int, default=2, choices=range(1, 9), help="Bit-width used for MoE/MLP projections.")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4", "mix"], help="Calibration dataset.")
    parser.add_argument("--eval_datasets", type=str, default="", help="Comma-separated datasets for PPL evaluation. Defaults to --dataset when omitted.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1, help="Replay batch size used during calibration and evaluation.")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--groupsize", type=int, default=128, help="Use -1 to disable grouping.")
    parser.add_argument("--sym", action="store_true", help="Use symmetric quantization.")
    parser.add_argument("--act-order", action="store_true", help="Enable the GPTQ activation-order heuristic.")
    parser.add_argument("--pack", action="store_true", help="Replace quantized linear layers with QLinear.")
    parser.add_argument("--save", action="store_true", help="Save the quantized model state after quantization.")
    parser.add_argument("--saving_path", type=str, default="", help="Output directory used together with --save.")
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate perplexity after quantization.")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated lm_eval task names.")
    parser.add_argument("--lm_eval_batch_size", type=str, default="auto", help="lm_eval batch size or 'auto'.")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--gen_kwargs", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation passed to transformers.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def get_model(args):
    disable_torch_init()
    config = AutoConfig.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(config, "model_type", None) not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={getattr(config, 'model_type', None)}. Expected one of {sorted(SUPPORTED_MODEL_TYPES)}."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    )
    model.seqlen = min(args.seqlen, getattr(config, "max_position_embeddings", args.seqlen))
    return model


def get_base_model(model):
    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "layers"):
        raise ValueError("Unsupported model layout: expected model.model.layers to exist.")
    return base_model


def move_if_present(module, device):
    if module is not None:
        return module.to(device)
    return None


def move_shared_modules(base_model, device):
    move_if_present(getattr(base_model, "embed_tokens", None), device)
    move_if_present(getattr(base_model, "norm", None), device)
    move_if_present(getattr(base_model, "rotary_emb", None), device)


def extract_hidden_states(outputs):
    # Qwen3 decoder layers return hidden states directly, while Qwen2-MoE returns a tuple.
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, tuple):
        return outputs[0]
    raise TypeError(f"Unexpected layer output type: {type(outputs)!r}")


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def batched_tensor_iterator(input_batches: Iterable[torch.Tensor], batch_size: int):
    buffer = []
    for tensor in input_batches:
        buffer.append(tensor)
        if len(buffer) == batch_size:
            yield torch.cat(buffer, dim=0)
            buffer = []
    if buffer:
        yield torch.cat(buffer, dim=0)


def adapt_batch_value(value, target_batch_size: int, source_batch_size: int):
    if torch.is_tensor(value):
        if value.dim() > 0 and value.shape[0] == source_batch_size and target_batch_size != source_batch_size:
            if target_batch_size < source_batch_size:
                return value[:target_batch_size]
            expand_shape = (target_batch_size,) + tuple(value.shape[1:])
            return value[:1].expand(*expand_shape)
        return value
    if isinstance(value, tuple):
        return tuple(adapt_batch_value(item, target_batch_size, source_batch_size) for item in value)
    if isinstance(value, list):
        return [adapt_batch_value(item, target_batch_size, source_batch_size) for item in value]
    if isinstance(value, dict):
        return {key: adapt_batch_value(item, target_batch_size, source_batch_size) for key, item in value.items()}
    return value


def get_layer_inputs(cache, batch_size: int):
    source_batch_size = cache["capture_batch_size"]
    layer_args = adapt_batch_value(cache["layer_args"], batch_size, source_batch_size)
    layer_kwargs = adapt_batch_value(cache["layer_kwargs"], batch_size, source_batch_size)
    return layer_args, layer_kwargs


def forward_decoder_layer(layer, hidden_states, layer_args, layer_kwargs):
    outputs = layer(hidden_states, *layer_args, **layer_kwargs)
    return extract_hidden_states(outputs)


def calibration_batches(dataloader: Iterable):
    for batch in dataloader:
        yield batch[0] if isinstance(batch, (tuple, list)) else batch


def test_batches(testenc, seqlen):
    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // seqlen
    for index in range(nsamples):
        start = index * seqlen
        end = (index + 1) * seqlen
        yield input_ids[:, start:end]


@torch.no_grad()
def collect_layer_inputs(model, input_batches: Iterable[torch.Tensor], nsamples: int, batch_size: int, device: str):
    base_model = get_base_model(model)
    layers = base_model.layers

    move_shared_modules(base_model, device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "layer_args": (), "layer_kwargs": {}, "capture_batch_size": 1}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            hidden_states = args[0]
            current_batch = hidden_states.shape[0]
            end = cache["i"] + current_batch
            inps[cache["i"]:end] = hidden_states
            cache["i"] = end
            cache["layer_args"] = tuple(args[1:])
            cache["layer_kwargs"] = dict(kwargs)
            cache["capture_batch_size"] = current_batch
            raise ValueError

    layers[0] = Catcher(layers[0])
    for input_ids in batched_tensor_iterator(input_batches, batch_size):
        if cache["i"] >= nsamples:
            break
        try:
            model(input_ids.to(device))
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    move_shared_modules(base_model, "cpu")
    torch.cuda.empty_cache()

    if cache["i"] != nsamples:
        raise ValueError(f"Expected {nsamples} calibration/eval samples, collected {cache['i']}.")

    return inps, cache


def classify_linear_layer(name: str):
    if ".self_attn." in name:
        return "attn"
    if ".mlp." not in name:
        return None
    if ".mlp.gate" in name or ".mlp.shared_expert_gate" in name:
        return "router"
    return "moe"


def resolve_module_path(root, module_name: str):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def replace_with_qlinear(layer, module_name: str, linear_layer: nn.Linear, scale, zero, bits: int, groupsize: int, device: str):
    quant_config = BaseQuantizeConfig(
        nbits=bits,
        group_size=None if groupsize == -1 else groupsize,
    )
    quant_layer = QLinear(quant_config=quant_config, device=device)
    if linear_layer.bias is not None:
        quant_layer.bias = linear_layer.bias.detach().to(device=device, dtype=torch.float16)
    quant_layer.replace_quantized_weight(linear_layer.weight.detach(), scale, zero)

    parent, attr = resolve_module_path(layer, module_name)
    setattr(parent, attr, quant_layer)


@torch.no_grad()
def qwen_sequential(model, dataloader, device: str, args):
    print("Starting GPTQ quantization ...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    base_model = get_base_model(model)
    layers = base_model.layers
    inps, cache = collect_layer_inputs(model, calibration_batches(dataloader), args.nsamples, args.batch_size, device)
    outs = torch.zeros_like(inps)

    stats = {"attn_modules": 0, "moe_modules": 0, "router_skipped": 0}

    for index in range(len(layers)):
        print(f"Quantizing layer {index + 1}/{len(layers)} ...")
        print("+--------------------------------+------------+------------+------------+---------+")
        print("|              name              |weight_error| fp_inp_SNR | q_inp_SNR  |  time   |")
        print("+================================+============+============+============+=========+")

        layer = layers[index].to(device)
        full = find_layers(layer)
        subset = {}

        for name, module in full.items():
            category = classify_linear_layer(name)
            if category == "router":
                stats["router_skipped"] += 1
                continue
            if category in {"attn", "moe"}:
                subset[name] = (module, category)

        gptq = {}
        for name, (module, category) in subset.items():
            bits = args.attn_bits if category == "attn" else args.moe_bits
            gptq[name] = GPTQ(module, LOGGER, name, bits)
            gptq[name].quantizer.configure(bits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
            gptq[name].wbits = bits
            stats[f"{category}_modules"] += 1
            print(f"Assign {bits}-bit to {name} ({category})")

        def add_batch(name):
            def tmp(_, inp, out):
                out_tensor = out[0] if isinstance(out, tuple) else out
                gptq[name].add_batch(inp[0].data, out_tensor.data)

            return tmp

        handles = [module.register_forward_hook(add_batch(name)) for name, (module, _) in subset.items()]

        for start in range(0, args.nsamples, args.batch_size):
            end = min(start + args.batch_size, args.nsamples)
            layer_args, layer_kwargs = get_layer_inputs(cache, end - start)
            outs[start:end] = forward_decoder_layer(
                layer,
                inps[start:end],
                layer_args,
                layer_kwargs,
            )

        for handle in handles:
            handle.remove()

        for name, (module, category) in subset.items():
            scale, zero, _, _ = gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=args.groupsize,
                actorder=args.act_order,
                name=name,
            )
            if args.pack:
                replace_with_qlinear(layer, name, module, scale, zero, gptq[name].wbits, args.groupsize, device)
            gptq[name].free()

        for start in range(0, args.nsamples, args.batch_size):
            end = min(start + args.batch_size, args.nsamples)
            layer_args, layer_kwargs = get_layer_inputs(cache, end - start)
            outs[start:end] = forward_decoder_layer(
                layer,
                inps[start:end],
                layer_args,
                layer_kwargs,
            )

        layers[index] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print("+--------------------------------+------------+------------+------------+---------+")
        print()

    model.config.use_cache = use_cache
    return stats


@torch.no_grad()
def evaluate_perplexity(model, testenc, device: str, dataset: str, batch_size: int):
    print(f"Evaluating perplexity on {dataset} ...")
    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // model.seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False

    base_model = get_base_model(model)
    layers = base_model.layers
    eval_batch_size = min(batch_size, nsamples) if nsamples > 0 else 1
    inps, cache = collect_layer_inputs(model, test_batches(testenc, model.seqlen), nsamples, eval_batch_size, device)
    outs = torch.zeros_like(inps)

    for index in range(len(layers)):
        print(index)
        layer = layers[index].to(device)
        for start in range(0, nsamples, eval_batch_size):
            end = min(start + eval_batch_size, nsamples)
            layer_args, layer_kwargs = get_layer_inputs(cache, end - start)
            outs[start:end] = forward_decoder_layer(
                layer,
                inps[start:end],
                layer_args,
                layer_kwargs,
            )
        layers[index] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    move_if_present(getattr(base_model, "norm", None), device)
    model.lm_head = model.lm_head.to(device)
    input_ids = input_ids.to(device)

    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    for index in range(nsamples):
        hidden_states = inps[index].unsqueeze(0)
        if getattr(base_model, "norm", None) is not None:
            hidden_states = base_model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, (index * model.seqlen): ((index + 1) * model.seqlen)][:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    model.config.use_cache = use_cache
    print(f"Perplexity: {ppl.item():.6f}")
    move_if_present(getattr(base_model, "norm", None), "cpu")
    model.lm_head = model.lm_head.cpu()
    return ppl.item()


def run_lm_eval(model, tokenizer, args):
    if args.tasks == "":
        return {}

    if not args.pack:
        LOGGER.warning("Running lm_eval without --pack keeps the model at fp16 weight size and may require substantially more GPU memory.")

    try:
        lm_eval = importlib.import_module("lm_eval")
        HFLM = importlib.import_module("lm_eval.models.huggingface").HFLM
    except ImportError as exc:
        raise ImportError("lm_eval is not installed. Install it before using --tasks.") from exc

    device = torch.device(get_device())
    model = model.to(device)
    eval_batch_size = args.lm_eval_batch_size or "auto"
    task_list = args.tasks.split(",") if isinstance(args.tasks, str) else args.tasks

    try:
        task_manager = lm_eval.tasks.TaskManager(
            include_path="./datasets_local/lm_eval_configs/tasks",
            include_defaults=True,
        )
    except Exception:
        task_manager = lm_eval.tasks.TaskManager(include_defaults=True)

    print(f"Initializing HFLM with batch_size={eval_batch_size} ...")
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size)

    task_results = lm_eval.simple_evaluate(
        model=hflm,
        tasks=task_list,
        batch_size=eval_batch_size,
        task_manager=task_manager,
        gen_kwargs=args.gen_kwargs,
        num_fewshot=args.num_fewshot,
    )["results"]

    metrics = {}
    for task, result in task_results.items():
        metrics[task] = round(result.get("acc_norm,none", result.get("acc,none", 0)), 4)

    LOGGER.info("Task Results: %s", metrics)
    pprint(metrics)
    return metrics


def maybe_save(model, tokenizer, args):
    if not args.save:
        return
    if not args.saving_path:
        raise ValueError("--saving_path is required when --save is set.")

    os.makedirs(args.saving_path, exist_ok=True)
    tokenizer.save_pretrained(args.saving_path)
    save_quantized(model, args.saving_path)
    print(f"Saved quantized model to {args.saving_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    print(f"Arguments: {args}")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")

    device = get_device()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workflow, but no CUDA device is available.")
    print(f"Using device {device} (controlled by CUDA_VISIBLE_DEVICES when CUDA is available)")

    model = get_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    start_time = time.time()
    stats = qwen_sequential(model, dataloader, device, args)
    print(f"Quantization time: {time.time() - start_time:.2f}s")
    print(f"Quantized attention modules: {stats['attn_modules']}")
    print(f"Quantized MoE modules: {stats['moe_modules']}")
    print(f"Skipped router modules: {stats['router_skipped']}")

    results = {}
    if args.eval_ppl:
        if args.eval_datasets:
            datasets = [item.strip() for item in args.eval_datasets.split(",") if item.strip()]
        elif args.dataset == "mix":
            datasets = ["wikitext2", "ptb", "c4"]
        else:
            datasets = [args.dataset]
        for dataset in datasets:
            _, testloader = get_loaders(dataset, seed=args.seed, seqlen=model.seqlen, model=args.model)
            if testloader is None:
                raise ValueError(f"Dataset '{dataset}' does not provide a test split for perplexity evaluation.")
            results[f"ppl/{dataset}"] = round(evaluate_perplexity(model, testloader, device, dataset, args.batch_size), 6)

    if args.tasks:
        results.update(run_lm_eval(model, tokenizer, args))

    maybe_save(model, tokenizer, args)

    if results:
        pprint(results)


if __name__ == "__main__":
    main()