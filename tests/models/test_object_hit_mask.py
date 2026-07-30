import torch

from hepattn.models.loss import mask_focal_loss
from hepattn.models.task import ObjectHitMaskTask


def test_object_hit_mask_focal_gamma_is_forwarded():
    task = ObjectHitMaskTask(
        name="mask",
        input_hit="node",
        input_object="query",
        output_object="pflow",
        target_object="particle",
        losses={"mask_focal": 5.0},
        costs={},
        dim=4,
        focal_gamma=3.0,
    )
    output = torch.tensor([[[0.5, -1.0, 2.0], [-0.5, 1.5, -2.0]]])
    target = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    targets = {
        "particle_node_valid": target,
        "node_valid": torch.tensor([[True, True, False]]),
        "particle_valid": torch.tensor([[True, False]]),
    }
    outputs = {"pflow_node_logit": output}

    expected = 5.0 * mask_focal_loss(
        output,
        target,
        gamma=3.0,
        object_valid_mask=targets["particle_valid"],
        input_pad_mask=targets["node_valid"],
        sample_weight=torch.ones_like(target),
    )

    torch.testing.assert_close(task.loss(outputs, targets)["mask_focal"], expected)
    torch.testing.assert_close(
        task.loss_per_element(outputs, targets)["mask_focal"],
        5.0
        * mask_focal_loss(
            output,
            target,
            gamma=3.0,
            object_valid_mask=targets["particle_valid"],
            input_pad_mask=targets["node_valid"],
            sample_weight=torch.ones_like(target),
            reduction="none",
        ),
    )
