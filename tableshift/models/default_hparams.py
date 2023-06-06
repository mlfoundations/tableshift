from fairlearn.reductions import ErrorRateParity
from frozendict import frozendict
from torch.nn import functional as F

from tableshift.core import TabularDataset
from tableshift.models.compat import is_pytorch_model_name
from tableshift.models.losses import GroupDROLoss, DROLoss

DEFAULT_BATCH_SIZE = 4096

# Default configs for testing models. These are not tuned
# or selected for any particular reason; they might not even
# be good choices for hyperparameters. These values do set
# values for the non-tuned hyperparameters (those not
# defined in the search space for each algorithm in
# tableshift.configs.default_hparams.py . Values here will *not* be overwritten
# with other defaults (e.g. if batch_size is specified for your model, it will
# not be set to the default batch size for the given model).

_DEFAULT_CONFIGS = frozendict({
    "aldro": {
        "num_layers": 2,
        "d_hidden": 512,
        "dropouts": 0.,
        "eta_pi": 0.01,
        "r": 1.,
    },
    "dann": {
        "num_layers": 2,
        "d_hidden": 256,
        "dropouts": 0.,
        'lr_d': 0.01,
        'weight_decay_d': 0.01,
        'lr_g': 0.01,
        'weight_decay_g': 0.01,
        'd_steps_per_g_step': 2,
        'grad_penalty': 0.01,
        'loss_lambda': 0.01,
    },
    "deepcoral":
        {"num_layers": 4,
         "d_hidden": 512,
         "mmd_gamma": 0.01,
         "dropouts": 0.},
    "dro":
        {"num_layers": 2,
         "d_hidden": 512,
         "dropouts": 0.,
         "geometry": "cvar",
         "size": 0.5,

         # Note: reg == 0 is equivalent to using chi-square constraint
         # (i.e. not using chi-square penalty).
         "reg": 0.,

         "max_iter": 10000},
    "expgrad":
        {"constraints": ErrorRateParity()},
    "ft_transformer":
        {"cat_cardinalities": None,
         "n_blocks": 1,
         "residual_dropout": 0.,
         "attention_dropout": 0.,
         "ffn_dropout": 0.,
         "ffn_factor": 1.,
         # This is feature embedding size in Table 13 above.
         "d_token": 64,
         },
    "group_dro":
        {"num_layers": 2,
         "d_hidden": 256,
         "group_weights_step_size": 0.05,
         "dropouts": 0.},
    "irm":
        {"num_layers": 2,
         "d_hidden": 256,
         "dropouts": 0.,
         "irm_lambda": 1e-2,
         # set irm_penalty_anneal_iters s.t. optimizer resets after 1 update
         "irm_penalty_anneal_iters": 1},
    "label_group_dro":
        {"num_layers": 2,
         "d_hidden": 256,
         "group_weights_step_size": 0.05,
         "dropouts": 0.},
    "mlp":
        {"num_layers": 2,
         "d_hidden": 256,
         "dropouts": 0.},
    "mixup":
        {"num_layers": 2,
         "d_hidden": 256,
         "dropouts": 0.,
         "mixup_alpha": 0.4},
    "mmd":
        {"num_layers": 4,
         "d_hidden": 512,
         "mmd_gamma": 0.01,
         "dropouts": 0.},
    "resnet":
        {"n_blocks": 2,
         "dropout_first": 0.2,
         "dropout_second": 0.,
         "hidden_factor": 1,
         "d_main": 128,
         "d_hidden": 256},
    "saint":
        {"dim": 8,
         # "depth": 6,
         # "heads": 8
         "depth": 1,
         "heads": 1,
         "attn_dropout": 0.1,
         "attentiontype": 'col',
         "ff_dropout": 0.1,
         },
    "node":
        {
            "batch_size": 256,
            "tree_dim": 1,
            "depth": 4,
            "num_layers": 2,
            "total_tree_count": 1024,
        },
    "tabtransformer":
        {"dim": 32,
         "depth": 6,
         "heads": 1,
         "attn_dropout": 0.1,
         "ff_dropout": 0.1,
         "mlp_hidden_mults": (4, 2),
         "batch_size": 256, },
    "vrex":
        {"num_layers": 4,
         "d_hidden": 512,
         "vrex_penalty_anneal_iters": 1,
         "vrex_lambda": 0.01,
         "dropouts": 0.},
})


def get_default_config(model: str, dset: TabularDataset) -> dict:
    """Get a default config for a model by name."""
    config = _DEFAULT_CONFIGS.get(model, {})

    d_in = dset.X_shape[1]
    if is_pytorch_model_name(model) and model != "ft_transformer":
        config.update({"d_in": d_in,
                       "activation": "ReLU"})
    elif is_pytorch_model_name(model):
        config.update({"n_num_features": d_in})

    if model in ("tabtransformer", "saint"):
        cat_idxs = dset.cat_idxs
        config["cat_idxs"] = cat_idxs
        config["categories"] = [2] * len(cat_idxs)

    # Models that use non-cross-entropy training objectives.
    if model == "dro":
        config["criterion"] = DROLoss(size=config["size"],
                                      reg=config["reg"],
                                      geometry=config["geometry"],
                                      max_iter=config["max_iter"])
    elif model == "group_dro":
        config["n_groups"] = dset.n_domains
        config["criterion"] = GroupDROLoss(n_groups=dset.n_domains)

    elif model == "label_group_dro":
        config["n_groups"] = 2  # assume binary classification
        config["criterion"] = GroupDROLoss(n_groups=2)


    else:
        config["criterion"] = F.binary_cross_entropy_with_logits

    if is_pytorch_model_name(model) and model != "dann":
        # Note: for DANN model, lr and weight decay are set separately for D
        # and G.
        config.update({"lr": 0.01,
                       "weight_decay": 0.01,
                       })

    # Do not overwrite batch size or epochs if they are set in the default
    # config for the model.
    if "batch_size" not in config:
        config["batch_size"] = DEFAULT_BATCH_SIZE
    if "n_epochs" not in config:
        config["n_epochs"] = 1

    if model == "saint" and d_in > 100:
        # same batch size setting logic as in SAINT code:
        # https://github.com/somepago/saint/blob/e288e84c77a54cfd2ffb55a53678fb7cbbb16630/train.py#LL95C5-L95C43
        config["batch_size"] = min(64, config["batch_size"])
    return config
