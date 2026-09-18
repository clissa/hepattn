"""Construct and invoke the Run-4 GLOW/GLOW-MDN models using hepattn definitions.

Keep this file, glow.yaml, glow-mdn.yaml and atlas_var_transform.yaml together.
See setup.md for the tensor contract and source modules to copy or import.
"""

import argparse
from pathlib import Path

import torch
import yaml
from jsonargparse import ArgumentParser

from hepattn.models import MaskFormer


def build_model(config_path: str | Path, attention: str = "torch") -> MaskFormer:
    """Instantiate the exact configured architecture, without a Lightning trainer.

    The default uses PyTorch SDPA in place of the trained FlashAttention encoder.
    This changes the backend, not the parameter layout. Use flash-varlen with
    CUDA autocast to exercise the original encoder backend.

    Raises:
        ValueError: If the attention backend is unsupported.
    """
    if attention not in {"torch", "flash-varlen"}:
        raise ValueError("attention must be torch or flash-varlen")
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    model_args = config["model"]["init_args"]
    model_args["encoder"]["init_args"]["attn_type"] = attention
    regression_args = model_args["tasks"]["init_args"]["modules"][-1]["init_args"]
    regression_args["scale_dict_path"] = str(config_path.parent / regression_args["scale_dict_path"])
    parser = ArgumentParser(exit_on_error=False)
    parser.add_subclass_arguments(MaskFormer, "model", required=True)
    return parser.instantiate_classes(parser.parse_object(config)).model


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: str = "cpu", attention: str = "torch") -> MaskFormer:
    """Load the model.* weights from a Lightning checkpoint, strictly.

    Raises:
        ValueError: If the checkpoint contains no model.* state entries.
    """
    model = build_model(config_path, attention=attention)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
    if not state:
        raise ValueError("Expected a Lightning checkpoint with model.* entries in state_dict")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def physical_particles(model: MaskFormer, final: dict) -> dict[str, torch.Tensor]:
    """Decode final slots to physical kinematics; preserve all 600 query slots.

    valid selects non-null classes only. Association masks and MDN distribution
    parameters remain available in the raw final outputs.
    """
    regression = next(task for task in model.tasks if task.name == "regression")
    scaled = final["regression"]["pflow_regr"]
    fields = {field: regression.scaler[field].inverse_transform(scaled[..., index]) for index, field in enumerate(regression.fields)}
    logits = final["classification"]["pflow_class_prob"]
    classes = logits.argmax(-1)
    return {
        **fields,
        "phi": torch.atan2(fields["sinphi"], fields["cosphi"]),
        "class": classes,
        "class_probabilities": logits.softmax(-1),
        "valid": classes < 5,
    }


def example_inputs(model: MaskFormer, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Make one synthetic event: two tracks, two topoclusters, four padding slots.

    This exercises tensor plumbing only; it is not an Athena event converter.
    """
    scaler = next(task for task in model.tasks if task.name == "regression").scaler
    features = torch.zeros(1, 8, 18, device=device)
    pt = torch.tensor([[10.0, 5.0, 8.0, 4.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    eta = torch.tensor([[0.2, -0.3, 0.2, -0.3, 0.0, 0.0, 0.0, 0.0]], device=device)
    phi = torch.tensor([[0.1, -0.2, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0]], device=device)
    valid = torch.arange(8, device=device).unsqueeze(0) < 4
    is_track = (torch.arange(8, device=device).unsqueeze(0) < 2).float()
    energy = pt * eta.cosh() * (1 - is_track)
    features[..., 0] = scaler["pt"].transform(pt)
    features[..., 1] = scaler["eta"].transform(eta)
    features[..., 2] = phi
    features[..., 3] = phi.cos()
    features[..., 4] = phi.sin()
    features[:, :2, 5:9] = features[:, :2, 1:5]
    features[:, 2:4, 11] = scaler["e"].transform(energy[:, 2:4])
    for column, field, value in [(12, "num_cells", 10.0), (13, "em_frac", 0.5), (14, "center_mag", 2000.0), (15, "center_lambda", 100.0)]:
        features[:, 2:4, column] = scaler[field].transform(torch.tensor(value, device=device))
    features[..., 16] = is_track
    features[..., 17] = valid.float() - is_track
    features.masked_fill_(~valid.unsqueeze(-1), 0.0)
    return {
        "node_features": features,
        "node_valid": valid,
        "node_e": energy,
        "node_pt": pt * is_track,
        "node_eta": eta,
        "node_phi": phi,
        "node_sinphi": phi.sin() * valid,
        "node_cosphi": phi.cos() * valid,
        "node_is_track": is_track,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention", choices=["torch", "flash-varlen"], default="torch")
    inputs_group = parser.add_mutually_exclusive_group(required=True)
    inputs_group.add_argument("--inputs", type=Path, help="torch.save tensor dictionary following setup.md")
    inputs_group.add_argument("--smoke-test", action="store_true", help="Use a synthetic event; no physics validation")
    parser.add_argument("--output", type=Path, help="Save raw final outputs and decoded particles")
    args = parser.parse_args()
    if args.attention == "flash-varlen" and torch.device(args.device).type != "cuda":
        parser.error("flash-varlen requires --device cuda (or cuda:N)")
    model = load_model(args.config, args.checkpoint, args.device, args.attention)
    inputs = (
        example_inputs(model, args.device)
        if args.smoke_test
        else {key: value.to(args.device) for key, value in torch.load(args.inputs, map_location="cpu", weights_only=True).items()}
    )
    with (
        torch.inference_mode(),
        torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.attention == "flash-varlen"),
    ):
        final = model(inputs)["final"]
        particles = physical_particles(model, final)
    for task, tensors in final.items():
        for name, tensor in tensors.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite output: {task}.{name}")
            print(f"{task}.{name}: {tuple(tensor.shape)} {tensor.dtype}")
    if args.output:
        torch.save(
            {
                "final": {task: {name: tensor.cpu() for name, tensor in tensors.items()} for task, tensors in final.items()},
                "particles": {name: tensor.cpu() for name, tensor in particles.items()},
            },
            args.output,
        )


if __name__ == "__main__":
    main()
