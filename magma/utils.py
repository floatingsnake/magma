import argparse
import torch.distributed as dist
from transformers import AutoTokenizer, GPT2TokenizerFast
import deepspeed
from pathlib import Path
import wandb
import os
import yaml
import torch
from collections import defaultdict
from torchtyping import TensorType
import gdown


def is_main():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def print_main(*msg):
    if is_main():
        print(*msg)


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    if dist.is_initialized():
        losses = losses.detach().clone()
        # We use `all_reduce` because it is better supported than `reduce`
        dist.all_reduce(losses, dist.ReduceOp.SUM)
        return losses / dist.get_world_size()
    else:
        return losses


def cycle(loader):
    while True:
        for data in loader:
            yield data


def get_tokenizer(name="gpt2", sequence_length=2048):
    """
    Gets tokenizer for LM
    """
    if name == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = sequence_length
        # setup lm settings
        tokenizer.add_special_tokens(
            {"cls_token": "<|image|>"}
        )  # add special image token to tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = sequence_length
        # setup lm settings
        tokenizer.add_special_tokens(
            {"cls_token": "<|image|>"}
        )  # add special image token to tokenizer
    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=False, help="path to your training config"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args.deepspeed = True
    return args


def wandb_log(*args, **kwargs):
    if is_main():
        wandb.log(*args, **kwargs)


def wandb_init(*args, **kwargs):
    if is_main():
        wandb.init(*args, **kwargs)


def save_model(model_engine, save_dir, global_step, config=None):
    os.makedirs(save_dir, exist_ok=True)
    if config is not None:
        config = config.to_dict()
        with open(str(Path(save_dir) / "config.yml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    sd = {"global_step": global_step, "config": config}
    model_engine.save_checkpoint(save_dir, client_state=sd)


def load_model(
    model_engine, load_dir, load_optimizer_states=True, load_lr_scheduler_states=True
):
    """
    Loads a model from disk and returns the global step to resume from if loading was successful, otherwise returns 0
    """
    try:
        load_path, sd = model_engine.load_checkpoint(
            load_dir,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
        )
    except AssertionError as e:
        load_path = None
        print(e)
    if load_path is None:
        print("Model loading failed - starting from global step 0")
        return 0
    return sd["global_step"]


def get_params_for_weight_decay_optimization(module, config):
    """
    Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and biases will have no weight decay but the rest will.
    """
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    blacklist_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    for module_ in module.modules():
        if isinstance(module_, blacklist_modules) or (
            config.weight_decay == 0.0
        ):  # also include all parameters here if no weight decay is being done
            no_weight_decay_params["params"].extend(
                [
                    p
                    for p in list(module_._parameters.values())
                    if (p is not None) and p.requires_grad
                ]
            )
        else:
            for n, p in list(module_._parameters.items()):
                if p is not None and p.requires_grad:
                    if n != "bias":
                        weight_decay_params["params"].append(p)
                    else:
                        no_weight_decay_params["params"].append(p)

    param_dict = {
        pn: p
        for pn, p in module.named_parameters()
        if p is not None and p.requires_grad
    }
    assert len(no_weight_decay_params["params"]) + len(
        weight_decay_params["params"]
    ) == len(
        param_dict.keys()
    ), "Number of params in both groups != total number of trainable params"
    if config.weight_decay == 0.0:
        # only return a single param group if no weight decay is being used anyway
        return [no_weight_decay_params]
    return [weight_decay_params, no_weight_decay_params]


def configure_param_groups(model, config):
    """
    Configures the different parameter groups in the model for training.
    If a separate learning rate for the image prefix is provided, we separate out the groups here.
    Additionally, parameters to which weight decay shouldn't be applied (layernorms / biases) are separated.
    """
    if config.image_enc_lr is not None:

        # get the params for the image prefix / proj
        image_enc_params = get_params_for_weight_decay_optimization(
            model.image_prefix.enc, config
        )
        for pdict in image_enc_params:
            pdict["lr"] = config.image_enc_lr
        image_proj_params = get_params_for_weight_decay_optimization(
            model.image_prefix.proj, config
        )

        # get the params for layernorm if it exists
        if config.use_image_embed_layernorm:
            image_ln_params = get_params_for_weight_decay_optimization(
                model.image_prefix.ln, config
            )
            image_proj_params += image_ln_params

        # get the params for the lm
        lm_params = get_params_for_weight_decay_optimization(model.lm, config)

        # get params for class head if it exists
        class_params = []
        if hasattr(model, "class_head") and model.class_head is not None:
            class_params = get_params_for_weight_decay_optimization(
                model.class_head, config
            )

        all_params = []
        for p in image_enc_params + lm_params + image_proj_params + class_params:
            if p["params"]:
                all_params.append(p)
    else:
        all_params = get_params_for_weight_decay_optimization(model, config)

    # merge param dicts with shared lr / wd values
    d = defaultdict(dict)
    for param_group in all_params:
        lr = param_group.get("lr", None)
        wd = param_group.get("weight_decay", None)
        key = f"lr_{lr}_wd_{wd}"
        if d[key].get("params") is None:
            d[key]["params"] = []
        d[key]["params"].extend(param_group["params"])
        if lr is not None:
            d[key]["lr"] = lr
        if wd is not None:
            d[key]["weight_decay"] = wd
    all_params = list(d.values())

    n_params = sum([len(d["params"]) for d in all_params])
    param_dict = {
        pn: p for pn, p in model.named_parameters() if p is not None and p.requires_grad
    }
    assert n_params == len(
        param_dict
    ), f"Some parameters are missing from param groups ({n_params} | {len(param_dict)})"

    # if we're using multiple param groups, set the min / max lr for each one[]
    # appropriately in deepspeed's scheduler
    config.deepspeed_config_params["scheduler"]["params"]["warmup_min_lr"] = [
        config.min_lr for _ in all_params
    ]
    config.deepspeed_config_params["scheduler"]["params"]["warmup_max_lr"] = [
        d.get("lr", config.lr) for d in all_params
    ]

    return all_params


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_table(name, model_outputs, gt_answers_list, global_step):
    results_table = wandb.Table(columns=["model output", "ground truth(s)"])
    for o, gt in zip(model_outputs, gt_answers_list):
        results_table.add_data(o, gt)
    wandb_log({f"eval/{name}": results_table}, step=global_step)

def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default
    
    
def get_world_info():
    local_rank = env2int(
        ['LOCAL_RANK', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK', 'SLURM_LOCALID'])
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)
    rank = env2int(['RANK', 'MPI_RANKID', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'SLURM_PROCID'])
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(rank)
    world_size = env2int(['WORLD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'SLURM_NPROCS'])
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)
    return local_rank, rank, world_size


def init_distributed(backend="nccl"):
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed(
            dist_backend=backend, verbose=True, auto_mpi_discovery=True
        )
    local_rank, rank, world_size = get_world_info()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def collate_fn_classification(batch_data, seq_len=2048):

    # for nvlr2: list(zip*(batch_data)) = [l_images, r_images, captions, class_labels]
    image_list = list(zip(*batch_data))[:-2]
    captions, class_labels = list(zip(*batch_data))[-2:]

    # images, captions, class_labels = list(zip(*batch_data))
    images_list = [torch.cat(image) for image in image_list]
    captions = torch.cat([i[:, :seq_len] for i in captions])
    class_labels = torch.stack(class_labels)
    return images_list, captions, class_labels


def infer_checkpoint_path_from_config(config):
    checkpoint_folder = config.save
    if checkpoint_folder is None:
        raise ValueError(
            "No checkpoint folder specified in config. Please provide a checkpoint."
        )

    # check for 'latest' tag in checkpoint folder
    if (Path(checkpoint_folder) / "latest").exists():
        latest_ckpt = (Path(checkpoint_folder) / "latest").read_text().strip()
    else:
        raise ValueError(
            f"No checkpoint found in {checkpoint_folder}. Please provide a checkpoint."
        )

    checkpoint_path = str(
        Path(checkpoint_folder) / latest_ckpt / "mp_rank_00_model_states.pt"
    )
    if not Path(checkpoint_path).exists():
        raise ValueError(
            f"No checkpoint found in {checkpoint_path}. Please provide a checkpoint."
        )

    return checkpoint_path


# [tensor_1, tensor_2], tensor_3, tensor_4 = to_cuda_half([tensor_1, tensor_2], tensor_3, tensor_4)
# probably not working yet
def to_cuda_half(*args):
    cuda_half_args = []
    for x in args:
        if isinstance(x, list):
            x_cuda_half = to_cuda_half(*x)
            cuda_half_args.append(x_cuda_half)
        elif isinstance(x, tuple):
            x_cuda_half = to_cuda_half(*x)
            cuda_half_args.append(x_cuda_half)
        else:
            if x.dtype in [torch.float32, torch.float16]:
                cuda_half_args.append(x.cuda().half())
            elif x.dtype == torch.long:
                cuda_half_args.append(x.cuda())

    if len(cuda_half_args) == 1:
        return cuda_half_args[0]
    else:
        return cuda_half_args


def build_labels(
    input_embeddings: TensorType["b", "s", "d"],
    captions: TensorType["b", "s"],
    eos_token,
    device,
) -> TensorType["b", "s"]:
    """
    Builds labels from input embeddings.

    Masks out the labels with -100 in positions up to the seq length of the embeddings, so loss is only computed for captions,
    and not for image tokens.
    Additionally, masks out everything *after* the first eos token.
    """
    shape = input_embeddings.shape[:2]  # b, s

    print(f'captions: {captions.shape[1]}')
    print(f'default: {shape[1]}')
    assert captions.shape[1] >= shape[1]

    # make sure to add masked embedding tokens in the appropriate locations in the labels
    embedding_tokens = torch.zeros(shape, dtype=torch.int64).to(device) - 100
    labels = torch.cat(
        (embedding_tokens, captions[:, : -shape[1]]), dim=1
    )  # we truncate the sequence length of the captions, as they are always padded to the full sequence length

    # mask out repeating eos tokens
    for label in labels:
        for k, token in enumerate(label):
            if token == eos_token:
                label[k + 1 :] = -100
                break

    return labels


def is_url(string):
    return string.startswith("http://") or string.startswith("https://")

def download_checkpoint(checkpoint_url, save_as):

    gdown.download(url = checkpoint_url, output = save_as, quiet=False)
