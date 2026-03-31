import torch

from nanochat import optim as optim_mod
from nanochat.optim import DistMuonAdamW, MuonAdamW


class _FakeFuture:
    def wait(self):
        return None


class _FakeWork:
    def get_future(self):
        return _FakeFuture()


def _install_fake_dist(monkeypatch):
    monkeypatch.setattr(optim_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(optim_mod.dist, "get_world_size", lambda: 1)

    def fake_all_reduce(tensor, op=None, async_op=False):
        return _FakeWork()

    def fake_reduce_scatter_tensor(output, input, op=None, async_op=False):
        output.copy_(input[: output.shape[0]])
        return _FakeWork()

    def fake_all_gather_into_tensor(output, input, async_op=False):
        output.copy_(input)
        return _FakeWork()

    monkeypatch.setattr(optim_mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(optim_mod.dist, "reduce_scatter_tensor", fake_reduce_scatter_tensor)
    monkeypatch.setattr(optim_mod.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)


def test_dist_adamw_skips_none_grads(monkeypatch):
    _install_fake_dist(monkeypatch)

    def fake_adamw_step(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
        p.add_(grad, alpha=-float(lr_t))

    monkeypatch.setattr(optim_mod, "adamw_step_fused", fake_adamw_step)

    p_active = torch.nn.Parameter(torch.ones(8))
    p_inactive = torch.nn.Parameter(torch.full((8,), 2.0))
    opt = DistMuonAdamW([
        dict(kind="adamw", params=[p_active, p_inactive], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    ])

    active_before = p_active.detach().clone()
    inactive_before = p_inactive.detach().clone()
    p_active.grad = torch.ones_like(p_active)
    p_inactive.grad = None

    opt.step()

    assert not torch.allclose(p_active, active_before)
    assert torch.allclose(p_inactive, inactive_before)
    assert "exp_avg" in opt.state[p_active]
    assert p_inactive not in opt.state or not opt.state[p_inactive]


def test_dist_muon_skips_none_grads(monkeypatch):
    _install_fake_dist(monkeypatch)
    call_shapes = []

    def fake_muon_step(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
        call_shapes.append(
            (
                tuple(stacked_grads.shape),
                tuple(stacked_params.shape),
                tuple(momentum_buffer.shape),
                tuple(second_momentum_buffer.shape),
            )
        )
        stacked_params.add_(stacked_grads, alpha=-float(lr_t))

    monkeypatch.setattr(optim_mod, "muon_step_fused", fake_muon_step)

    p_active = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).view(4, 4) / 16)
    p_inactive = torch.nn.Parameter(torch.eye(4, dtype=torch.float32))
    opt = DistMuonAdamW([
        dict(kind="muon", params=[p_active, p_inactive], lr=0.01, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0)
    ])

    active_before = p_active.detach().clone()
    inactive_before = p_inactive.detach().clone()
    p_active.grad = torch.full_like(p_active, 0.25)
    p_inactive.grad = None

    opt.step()

    assert not torch.allclose(p_active, active_before)
    assert torch.allclose(p_inactive, inactive_before)
    assert "momentum_buffer" in opt.state[p_active]
    assert opt.state[p_active]["momentum_buffer"].shape[0] == 2
    assert call_shapes == [((2, 4, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1))]


def test_single_gpu_muon_skips_none_grads(monkeypatch):
    call_shapes = []

    def fake_muon_step(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer, momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
        call_shapes.append(
            (
                tuple(stacked_grads.shape),
                tuple(stacked_params.shape),
                tuple(momentum_buffer.shape),
                tuple(second_momentum_buffer.shape),
            )
        )
        stacked_params.add_(stacked_grads, alpha=-float(lr_t))

    monkeypatch.setattr(optim_mod, "muon_step_fused", fake_muon_step)

    p_active = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).view(4, 4) / 16)
    p_inactive = torch.nn.Parameter(torch.eye(4, dtype=torch.float32))
    opt = MuonAdamW([
        dict(kind="muon", params=[p_active, p_inactive], lr=0.01, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0)
    ])

    active_before = p_active.detach().clone()
    inactive_before = p_inactive.detach().clone()
    p_active.grad = torch.full_like(p_active, 0.25)
    p_inactive.grad = None

    opt.step()

    assert not torch.allclose(p_active, active_before)
    assert torch.allclose(p_inactive, inactive_before)
    assert "momentum_buffer" in opt.state[p_active]
    assert opt.state[p_active]["momentum_buffer"].shape[0] == 2
    assert call_shapes == [((2, 4, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1))]
