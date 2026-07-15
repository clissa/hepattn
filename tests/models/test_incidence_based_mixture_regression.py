import math

import pytest
import torch
from torch import nn

from hepattn.models.task import IncidenceBasedMixtureRegressionTask


class ConstantHead(nn.Module):
    def __init__(self, values: list[float]):
        super().__init__()
        self.output_size = len(values)
        self.register_buffer("values", torch.tensor(values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values.expand(*x.shape[:-1], -1)


def make_task(
    tmp_path,
    head_values: list[float],
    *,
    num_components: int = 1,
    mdn_fields: list[str] | None = None,
    deterministic_fields: list[str] | None = None,
    mean_mode: str = "offset",
    scale_floor: float = 1.0e-3,
    initial_scale: float = 1.0e-1,
) -> IncidenceBasedMixtureRegressionTask:
    scale_path = tmp_path / "scales.yaml"
    scale_path.write_text("{}\n")
    return IncidenceBasedMixtureRegressionTask(
        name="regression",
        input_hit="node",
        input_object="query",
        output_object="pflow",
        target_object="particle",
        scale_dict_path=str(scale_path),
        net=ConstantHead(head_values),
        cost_weight=10.0,
        use_nodes=False,
        has_intermediate_loss=False,
        mdn_fields=mdn_fields or ["e", "pt"],
        deterministic_fields=deterministic_fields or ["eta", "sinphi", "cosphi"],
        num_components=num_components,
        mean_mode=mean_mode,
        scale_floor=scale_floor,
        initial_scale=initial_scale,
    )


def stub_proxy(task: IncidenceBasedMixtureRegressionTask, proxy: torch.Tensor) -> None:
    is_charged = torch.ones(proxy.shape[:-1], dtype=torch.bool)

    def get_proxy_feats(_incidence, _inputs, **_kwargs):
        return proxy, is_charged, (proxy + 100, proxy + 200)

    task.get_proxy_feats = get_proxy_feats


def forward_inputs(batch_size: int = 2, num_queries: int = 3) -> dict[str, torch.Tensor]:
    return {
        "query_embed": torch.zeros(batch_size, num_queries, 4),
        "incidence": torch.zeros(batch_size, num_queries, 5),
        "class_probs": torch.zeros(batch_size, num_queries, 6),
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_components": 0}, "num_components"),
        ({"mdn_fields": ["e", "pt"], "deterministic_fields": ["pt", "eta", "sinphi"]}, "disjoint"),
        ({"mdn_fields": ["pt", "e"], "deterministic_fields": ["eta", "sinphi", "cosphi"]}, "field order"),
        ({"mean_mode": "scale"}, "mean_mode"),
        ({"scale_floor": 0.0}, "scale_floor"),
        ({"scale_floor": 0.1, "initial_scale": 0.1}, "initial_scale"),
    ],
)
def test_constructor_validation(tmp_path, kwargs, match):
    defaults = {
        "num_components": 1,
        "mdn_fields": ["e", "pt"],
        "deterministic_fields": ["eta", "sinphi", "cosphi"],
        "mean_mode": "offset",
        "scale_floor": 1.0e-3,
        "initial_scale": 1.0e-1,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        make_task(tmp_path, [0.0] * 8, **defaults)


def test_constructor_validates_head_width(tmp_path):
    with pytest.raises(ValueError, match="output_size"):
        make_task(tmp_path, [0.0] * 7)


def test_constructor_validates_input_width_when_embedding_dim_is_set(tmp_path):
    class Head(nn.Module):
        input_size = 518
        output_size = 8

        def forward(self, x):
            return x

    scale_path = tmp_path / "scales.yaml"
    scale_path.write_text("{}\n")
    with pytest.raises(ValueError, match=r"must be 390, got 518.*5 proxy features.*1 charged flag"):
        IncidenceBasedMixtureRegressionTask(
            name="regression",
            input_hit="node",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            scale_dict_path=str(scale_path),
            net=Head(),
            cost_weight=10.0,
            use_nodes=True,
            embedding_dim=192,
            has_intermediate_loss=False,
        )


@pytest.mark.parametrize(("num_components", "width"), [(1, 8), (3, 18)])
def test_head_shapes_and_log_weight_normalization(tmp_path, num_components, width):
    task = make_task(tmp_path, [0.0] * width, num_components=num_components)
    proxy = torch.zeros(2, 3, 5)
    stub_proxy(task, proxy)

    outputs = task(forward_inputs())

    assert outputs["pflow_mdn_log_weights"].shape == (2, 3, num_components)
    assert outputs["pflow_mdn_means"].shape == (2, 3, num_components, 2)
    assert outputs["pflow_mdn_scales"].shape == (2, 3, num_components, 2)
    assert outputs["pflow_deterministic_regr"].shape == (2, 3, 3)
    assert torch.isfinite(outputs["pflow_mdn_log_weights"]).all()
    torch.testing.assert_close(outputs["pflow_mdn_log_weights"].logsumexp(-1), torch.zeros(2, 3))


def test_scale_floor_and_inverse_softplus_initialization(tmp_path):
    task = make_task(tmp_path, [0.0] * 8, scale_floor=0.01, initial_scale=0.2)
    stub_proxy(task, torch.zeros(2, 3, 5))

    scales = task(forward_inputs())["pflow_mdn_scales"]

    assert torch.all(scales > task.scale_floor)
    torch.testing.assert_close(scales, torch.full_like(scales, 0.2))
    expected_offset = math.log(math.expm1(0.2 - 0.01))
    torch.testing.assert_close(task.scale_offset, torch.tensor(expected_offset))


@pytest.mark.parametrize(
    ("mean_mode", "expected_point"),
    [
        ("offset", [11.0, 22.0, 33.0, 44.0, 55.0]),
        ("absolute", [1.0, 2.0, 3.0, 4.0, 5.0]),
    ],
)
def test_offset_and_absolute_point_predictions(tmp_path, mean_mode, expected_point):
    task = make_task(tmp_path, [0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0], mean_mode=mean_mode)
    proxy = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]).expand(2, 3, -1)
    stub_proxy(task, proxy)

    outputs = task(forward_inputs())

    torch.testing.assert_close(outputs["pflow_regr"], torch.tensor(expected_point).expand(2, 3, -1))


def test_scaled_space_mixture_expectation(tmp_path):
    logits = [0.0, math.log(3.0)]
    means = [1.0, 2.0, 5.0, 10.0]
    raw_scales = [0.0] * 4
    deterministic = [3.0, 4.0, 5.0]
    task = make_task(tmp_path, logits + means + raw_scales + deterministic, num_components=2, mean_mode="absolute")
    stub_proxy(task, torch.zeros(2, 3, 5))

    point = task(forward_inputs())["pflow_regr"]

    torch.testing.assert_close(point, torch.tensor([4.0, 8.0, 3.0, 4.0, 5.0]).expand(2, 3, -1))


def loss_outputs(
    log_weights: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    deterministic: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "pflow_mdn_log_weights": log_weights,
        "pflow_mdn_means": means,
        "pflow_mdn_scales": scales,
        "pflow_deterministic_regr": deterministic,
    }


def loss_targets(mdn: torch.Tensor, deterministic: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "particle_e": mdn[..., 0],
        "particle_pt": mdn[..., 1],
        "particle_eta": deterministic[..., 0],
        "particle_sinphi": deterministic[..., 1],
        "particle_cosphi": deterministic[..., 2],
        "particle_valid": valid,
    }


def test_analytic_mixture_nll(tmp_path):
    task = make_task(tmp_path, [0.0] * 13, num_components=2, mean_mode="absolute")
    log_weights = torch.log(torch.tensor([[[0.25, 0.75]]]))
    means = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    scales = torch.ones_like(means)
    deterministic = torch.zeros(1, 1, 3)
    targets = loss_targets(torch.zeros(1, 1, 2), deterministic, torch.ones(1, 1, dtype=torch.bool))

    losses = task.loss(loss_outputs(log_weights, means, scales, deterministic), targets)

    expected = -math.log((0.25 + 0.75 * math.exp(-1.0)) / (2 * math.pi))
    torch.testing.assert_close(losses["mdn_nll"], torch.tensor(expected))
    torch.testing.assert_close(losses["deterministic_l1"], torch.tensor(0.0))


def test_single_component_matches_diagonal_gaussian_nll(tmp_path):
    task = make_task(tmp_path, [0.0] * 8, mean_mode="absolute")
    means = torch.tensor([[[[0.5, -0.5]]]])
    scales = torch.tensor([[[[0.7, 1.3]]]])
    target_mdn = torch.tensor([[[1.0, 2.0]]])
    deterministic = torch.zeros(1, 1, 3)
    targets = loss_targets(target_mdn, deterministic, torch.ones(1, 1, dtype=torch.bool))

    losses = task.loss(loss_outputs(torch.zeros(1, 1, 1), means, scales, deterministic), targets)

    expected = -torch.distributions.Independent(torch.distributions.Normal(means.squeeze(-2), scales.squeeze(-2)), 1).log_prob(target_mdn)
    torch.testing.assert_close(losses["mdn_nll"], expected.mean())


def test_extreme_mixture_values_remain_finite(tmp_path):
    task = make_task(tmp_path, [0.0] * 13, num_components=2, mean_mode="absolute")
    log_weights = torch.log_softmax(torch.tensor([[[-1000.0, 1000.0]]]), dim=-1)
    means = torch.tensor([[[[-1.0e4, 1.0e4], [1.0e4, -1.0e4]]]])
    scales = torch.tensor([[[[1.0e-3, 1.0e3], [1.0e3, 1.0e-3]]]])
    deterministic = torch.tensor([[[1.0e4, -1.0e4, 1.0e4]]])
    targets = loss_targets(torch.zeros(1, 1, 2), torch.zeros(1, 1, 3), torch.ones(1, 1, dtype=torch.bool))

    losses = task.loss(loss_outputs(log_weights, means, scales, deterministic), targets)

    assert all(torch.isfinite(loss) for loss in losses.values())


def test_valid_masking_and_weighted_separate_losses(tmp_path):
    task = make_task(tmp_path, [0.0] * 8, mean_mode="absolute")
    means = torch.tensor([[[[0.0, 0.0]], [[100.0, 100.0]], [[2.0, 2.0]]]])
    scales = torch.ones_like(means)
    deterministic = torch.tensor([[[1.0, 1.0, 1.0], [100.0, 100.0, 100.0], [3.0, 3.0, 3.0]]])
    target_mdn = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]])
    target_deterministic = torch.zeros(1, 3, 3)
    valid = torch.tensor([[True, False, True]])
    targets = loss_targets(target_mdn, target_deterministic, valid)

    losses = task.loss(loss_outputs(torch.zeros(1, 3, 1), means, scales, deterministic), targets)

    expected_nll = torch.tensor([math.log(2 * math.pi), math.log(2 * math.pi) + 1.0]).mean()
    torch.testing.assert_close(losses["mdn_nll"], expected_nll)
    torch.testing.assert_close(losses["deterministic_l1"], torch.tensor(12.0))


def test_empty_valid_batch_returns_differentiable_zeros(tmp_path):
    task = make_task(tmp_path, [0.0] * 8, mean_mode="absolute")
    log_weights = torch.zeros(1, 2, 1, requires_grad=True)
    means = torch.zeros(1, 2, 1, 2, requires_grad=True)
    scales = torch.ones(1, 2, 1, 2, requires_grad=True)
    deterministic = torch.zeros(1, 2, 3, requires_grad=True)
    targets = loss_targets(torch.zeros(1, 2, 2), torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.bool))

    losses = task.loss(loss_outputs(log_weights, means, scales, deterministic), targets)
    sum(losses.values()).backward()

    assert losses.keys() == {"mdn_nll", "deterministic_l1"}
    assert all(loss.item() == 0.0 and loss.requires_grad for loss in losses.values())
    assert all(tensor.grad is not None for tensor in (log_weights, means, scales, deterministic))


def test_loss_per_element_shapes_and_invalid_zeros(tmp_path):
    task = make_task(tmp_path, [0.0] * 8, mean_mode="absolute")
    means = torch.zeros(2, 3, 1, 2)
    scales = torch.ones_like(means)
    deterministic = torch.ones(2, 3, 3)
    valid = torch.tensor([[True, False, True], [False, True, False]])
    targets = loss_targets(torch.zeros(2, 3, 2), torch.zeros(2, 3, 3), valid)

    losses = task.loss_per_element(loss_outputs(torch.zeros(2, 3, 1), means, scales, deterministic), targets)

    assert losses["mdn_nll"].shape == (2, 3)
    assert losses["deterministic_l1"].shape == (2, 3)
    assert torch.equal(losses["mdn_nll"][~valid], torch.zeros(3))
    assert torch.equal(losses["deterministic_l1"][~valid], torch.zeros(3))


def test_single_component_mean_and_scale_gradients(tmp_path):
    task = make_task(tmp_path, [0.0] * 8, mean_mode="absolute")
    means = torch.zeros(1, 1, 1, 2, requires_grad=True)
    scales = torch.full((1, 1, 1, 2), 2.0, requires_grad=True)
    deterministic = torch.zeros(1, 1, 3, requires_grad=True)
    targets = loss_targets(torch.ones(1, 1, 2), torch.ones(1, 1, 3), torch.ones(1, 1, dtype=torch.bool))

    losses = task.loss(loss_outputs(torch.zeros(1, 1, 1), means, scales, deterministic), targets)
    sum(losses.values()).backward()

    assert torch.count_nonzero(means.grad) == means.numel()
    assert torch.count_nonzero(scales.grad) == scales.numel()
    assert torch.count_nonzero(deterministic.grad) == deterministic.numel()


def test_non_symmetric_mixture_has_logit_gradients(tmp_path):
    task = make_task(tmp_path, [0.0] * 13, num_components=2, mean_mode="absolute")
    logits = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    means = torch.tensor([[[[0.0, 0.0], [3.0, 3.0]]]], requires_grad=True)
    scales = torch.ones_like(means, requires_grad=True)
    deterministic = torch.zeros(1, 1, 3, requires_grad=True)
    targets = loss_targets(torch.zeros(1, 1, 2), deterministic.detach(), torch.ones(1, 1, dtype=torch.bool))

    losses = task.loss(loss_outputs(torch.log_softmax(logits, dim=-1), means, scales, deterministic), targets)
    losses["mdn_nll"].backward()

    assert torch.count_nonzero(logits.grad) == logits.numel()


def test_pairwise_matching_cost_is_detached_float32_five_field_l1(tmp_path):
    task = make_task(tmp_path, [0.0] * 8)
    point = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    targets = {
        "particle_e": torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64),
        "particle_pt": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        "particle_eta": torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64),
        "particle_sinphi": torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float64),
        "particle_cosphi": torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float64),
    }

    cost = task.cost({"pflow_regr": point}, targets)["regr_l1"]
    target = torch.stack([targets[f"particle_{field}"] for field in task.fields], dim=-1).float()
    expected = 10.0 * (point.detach().float().unsqueeze(2) - target.unsqueeze(1)).abs().mean(dim=-1)

    assert cost.shape == (1, 2, 3)
    assert cost.dtype == torch.float32
    assert not cost.requires_grad
    torch.testing.assert_close(cost, expected)


def test_all_forward_outputs_support_matching_permutation(tmp_path):
    task = make_task(tmp_path, [0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0])
    stub_proxy(task, torch.zeros(2, 3, 5))
    outputs = task(forward_inputs())

    assert set(outputs) == set(task.outputs)
    assert all(value.shape[:2] == (2, 3) for value in outputs.values())

    pred_idxs = torch.tensor([[2, 0, 1], [1, 2, 0]])
    batch_idxs = torch.arange(2).unsqueeze(1)
    permuted = {name: value[batch_idxs, pred_idxs] for name, value in outputs.items()}
    targets = loss_targets(torch.zeros(2, 3, 2), torch.zeros(2, 3, 3), torch.ones(2, 3, dtype=torch.bool))

    losses = task.loss(permuted, targets)

    assert all(torch.isfinite(loss) for loss in losses.values())


def test_predict_preserves_legacy_schema_without_mixture_parameters(tmp_path):
    task = make_task(tmp_path, [0.0] * 8)
    stub_proxy(task, torch.zeros(2, 3, 5))

    predictions = task.predict(task(forward_inputs()))

    fields = ["e", "pt", "eta", "sinphi", "cosphi"]
    expected = (
        {f"pflow_{field}" for field in fields}
        | {f"pflow_proxy_{field}" for field in fields}
        | {f"pflow_proxy_ch_{field}" for field in fields}
        | {f"pflow_proxy_neut_{field}" for field in fields}
        | {"pflow_is_charged"}
    )
    assert set(predictions) == expected
    assert not any("mdn" in name or "mixture" in name for name in predictions)
