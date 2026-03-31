import torch

from nanochat.gpt import GPT, GPTConfig


def _build_model(depth, n_embd=128, n_head=1):
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    return model


@torch.no_grad()
def _fill_state_with_arange(model):
    state = model.state_dict()
    for tensor in state.values():
        if not torch.is_floating_point(tensor):
            continue
        values = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
        tensor.copy_(values.to(device=tensor.device, dtype=tensor.dtype))
    return {name: tensor.clone() for name, tensor in state.items()}


def test_gpt_forward_only_executes_active_layers_spread_across_stack():
    model = _build_model(depth=4)
    model.set_active_depth(2)
    calls = [0, 0, 0, 0]
    hooks = []

    for idx, block in enumerate(model.transformer.h):
        def _hook(_module, _args, _output, layer_idx=idx):
            calls[layer_idx] += 1
        hooks.append(block.register_forward_hook(_hook))

    try:
        idx = torch.randint(0, model.config.vocab_size, (2, 8))
        model(idx)
    finally:
        for hook in hooks:
            hook.remove()

    assert model.get_active_layer_indices() == (0, 3)
    assert calls == [1, 0, 0, 1]


def test_gpt_expand_active_depth_fills_gaps_in_whole_stack():
    model = _build_model(depth=4)
    state = _fill_state_with_arange(model)
    model.load_state_dict(state)
    model.set_active_depth(2)

    summary = model.expand_active_depth(4)
    after = model.state_dict()

    assert model.get_active_depth() == 4
    assert model.get_active_layer_indices() == (0, 1, 2, 3)
    assert summary["source_depth"] == 2
    assert summary["target_depth"] == 4
    assert summary["source_active_layers"] == [0, 3]
    assert summary["target_active_layers"] == [0, 1, 2, 3]
    assert summary["layer_map"] == {1: 0, 2: 3}
    assert summary["copied_blocks"] == 2
    assert summary["copied_value_embeds"] == {}

    torch.testing.assert_close(after["transformer.h.1.attn.c_q.weight"], state["transformer.h.0.attn.c_q.weight"])
    torch.testing.assert_close(after["transformer.h.2.attn.c_q.weight"], state["transformer.h.3.attn.c_q.weight"])
    torch.testing.assert_close(after["transformer.h.1.mlp.c_fc.weight"], state["transformer.h.0.mlp.c_fc.weight"])
    torch.testing.assert_close(after["transformer.h.2.mlp.c_fc.weight"], state["transformer.h.3.mlp.c_fc.weight"])
    torch.testing.assert_close(after["resid_lambdas"], state["resid_lambdas"][torch.tensor([0, 0, 3, 3])])
    torch.testing.assert_close(after["x0_lambdas"], state["x0_lambdas"][torch.tensor([0, 0, 3, 3])])


def test_gpt_expand_active_depth_can_ramp_new_layer_outputs():
    model = _build_model(depth=4)
    state = _fill_state_with_arange(model)
    model.load_state_dict(state)
    model.set_active_depth(2)

    summary = model.expand_active_depth(4, growth_step=10, new_layer_scale=0.25, ramp_steps=20)

    assert summary["current_scale"] == 0.25
    torch.testing.assert_close(
        model.layer_output_scales,
        torch.tensor([1.0, 0.25, 0.25, 1.0], dtype=model.layer_output_scales.dtype),
    )

    current_scale = model.update_active_depth_growth_scale(20)
    assert abs(current_scale - 0.625) < 1e-6
    restored = model.restore_active_depth_growth(summary, current_step=15)
    assert restored is not None
    assert abs(restored["current_scale"] - 0.4375) < 1e-6

    current_scale = model.update_active_depth_growth_scale(30)
    assert current_scale == 1.0
    assert model.get_active_depth_growth_state() is None
    torch.testing.assert_close(model.layer_output_scales, torch.ones_like(model.layer_output_scales))


def test_gpt_expand_active_depth_copy_zero_l_zeros_new_output_projections_only():
    model = _build_model(depth=4)
    state = _fill_state_with_arange(model)
    model.load_state_dict(state)
    model.set_active_depth(2)

    summary = model.expand_active_depth(4, new_layer_init="copy-zeroL")
    after = model.state_dict()

    assert summary["new_layer_init"] == "copy-zeroL"
    assert summary["zeroed_last_linears"] == [1, 2]
    assert torch.count_nonzero(after["transformer.h.1.attn.c_proj.weight"]) == 0
    assert torch.count_nonzero(after["transformer.h.1.mlp.c_proj.weight"]) == 0
    assert torch.count_nonzero(after["transformer.h.2.attn.c_proj.weight"]) == 0
    assert torch.count_nonzero(after["transformer.h.2.mlp.c_proj.weight"]) == 0
