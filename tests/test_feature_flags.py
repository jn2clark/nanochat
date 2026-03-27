import math

import torch

from nanochat.gpt import EmbeddingLinear, GPT, GPTConfig


def build_model(**overrides):
    config = GPTConfig(
        sequence_len=256,
        vocab_size=128,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        **overrides,
    )
    return GPT(config, pad_vocab_size_to=1)


def test_master_dense_defaults_surface():
    with torch.device("meta"):
        model = build_model()

    assert model.config.use_smear is True
    assert model.config.use_backout is True
    assert model.config.use_value_embeds is True
    assert model.bigram is None
    assert model.backout_lambda is not None
    assert set(model.value_embeds.keys()) == {"1", "3"}
    assert all(block.attn.attn_gate is None for block in model.transformer.h)
    assert model.window_sizes[0] == (128, 0)
    assert model.window_sizes[-1] == (256, 0)


def test_disable_dense_features_removes_modules():
    with torch.device("meta"):
        model = build_model(use_smear=False, use_backout=False, use_value_embeds=False)

    assert model.smear_gate is None
    assert model.smear_lambda is None
    assert model.backout_lambda is None
    assert len(model.value_embeds) == 0
    assert all(block.attn.ve_gate is None for block in model.transformer.h)


def test_c7_feature_surface_builds_expected_modules():
    with torch.device("meta"):
        model = build_model(
            use_sparse_funnel=True,
            n_global=2,
            global_layer_override=(1, 3),
            local_window=128,
            local_mlp_ratio=4,
            rope_base=200000,
            bigram_vocab_size=4096,
            bigram_dim=32,
            use_gated_attn=True,
            ve_layers=(1, 3),
            init_profile="c7",
            optimizer_profile="c7",
            use_embedding_lm_head=True,
        )

    assert model._global_layers == (1, 3)
    assert model.window_sizes == [(128, 0), (256, 0), (128, 0), (256, 0)]
    assert model.bigram is not None
    assert isinstance(model.lm_head, EmbeddingLinear)
    assert all(block.attn.attn_gate is not None for block in model.transformer.h)
    assert set(model.value_embeds.keys()) == {"1", "3"}


def test_sparse_value_embeds_default_to_global_layers():
    with torch.device("meta"):
        model = build_model(
            use_sparse_funnel=True,
            n_global=2,
            global_layer_override=(1, 3),
        )

    assert model.config.ve_layers == (1, 3)
    assert set(model.value_embeds.keys()) == {"1", "3"}


def test_init_profiles_change_scalars_and_projection_init():
    master_model = build_model(init_profile="master")
    master_model.init_weights()
    c7_model = build_model(init_profile="c7")
    c7_model.init_weights()

    master_resid = torch.tensor([1.15, 1.1166667, 1.0833334, 1.05], dtype=master_model.resid_lambdas.dtype)
    master_x0 = torch.tensor([0.20, 0.15, 0.10, 0.05], dtype=master_model.x0_lambdas.dtype)
    assert torch.allclose(master_model.resid_lambdas, master_resid, atol=1e-6)
    assert torch.allclose(master_model.x0_lambdas, master_x0, atol=1e-6)
    assert torch.count_nonzero(master_model.transformer.h[0].attn.c_proj.weight) == 0

    resid_start, resid_end = 1.18, 1.06
    resid_decay = math.log(resid_start / resid_end) / 3
    c7_resid = torch.tensor(
        [resid_start * math.exp(-resid_decay * i) for i in range(4)],
        dtype=c7_model.resid_lambdas.dtype,
    )
    c7_x0 = torch.tensor([0.24, 0.08, 0.0, 0.0], dtype=c7_model.x0_lambdas.dtype)
    assert torch.allclose(c7_model.resid_lambdas, c7_resid, atol=1e-6)
    assert torch.allclose(c7_model.x0_lambdas, c7_x0, atol=1e-6)
    assert torch.count_nonzero(c7_model.transformer.h[0].attn.c_proj.weight) > 0


def test_optimizer_profile_changes_backout_lr():
    master_model = build_model(
        bigram_vocab_size=4096,
        bigram_dim=32,
        use_gated_attn=True,
        optimizer_profile="master",
    )
    c7_model = build_model(
        bigram_vocab_size=4096,
        bigram_dim=32,
        use_gated_attn=True,
        optimizer_profile="c7",
    )

    master_opt = master_model.setup_optimizer()
    c7_opt = c7_model.setup_optimizer()

    master_lrs = {id(param): group["lr"] for group in master_opt.param_groups for param in group["params"]}
    c7_lrs = {id(param): group["lr"] for group in c7_opt.param_groups for param in group["params"]}

    assert master_lrs[id(master_model.backout_lambda)] == 0.2
    assert c7_lrs[id(c7_model.backout_lambda)] == 0.15
    assert c7_lrs[id(c7_model.bigram.scale)] == 0.1
    assert c7_lrs[id(c7_model.transformer.h[0].attn.attn_gate.weight)] == 0.15


def test_master_optimizer_keeps_smear_and_backout_together():
    model = build_model(optimizer_profile="master")
    optimizer = model.setup_optimizer()

    target_ids = {id(model.smear_gate.weight), id(model.smear_lambda), id(model.backout_lambda)}
    matched_group = None
    for group in optimizer.param_groups:
        param_ids = {id(param) for param in group["params"]}
        if target_ids.issubset(param_ids):
            matched_group = group
            break

    assert matched_group is not None
    assert matched_group["kind"] == "adamw"
    assert matched_group["lr"] == 0.2
