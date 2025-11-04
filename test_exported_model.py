"""Test script to verify exported JAX model works correctly."""

import argparse
from pathlib import Path

import jax.numpy as jnp
from jax import export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the .jax_export model file")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist")

    # Load the exported model
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        exported = export.deserialize(f.read())

    # Create example inputs
    num_joints = 16  # Zeroth robot without foot sensors
    joint_angles = jnp.zeros(num_joints)
    joint_vels = jnp.zeros(num_joints)
    quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    command = jnp.zeros(7)  # [vx, vy, wz, heading, bh, rx, ry]
    carry = jnp.zeros((5, 192))  # (depth, hidden_size)

    # Run inference
    print("\nRunning inference...")
    action, next_carry = exported.call(joint_angles, joint_vels, quat, command, carry)

    # Display results
    print(f"\nInference successful!")
    print(f"  Action shape: {action.shape}")
    print(f"  Action values: {action}")
    print(f"  Next carry shape: {next_carry.shape}")
    print(f"\nModel is working correctly!")


if __name__ == "__main__":
    main()
