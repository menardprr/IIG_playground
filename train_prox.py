from datetime import datetime
import os
from agents.prox import ProxAgent


import yaml
import argparse
import gc
import jax

# see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

if __name__ == '__main__':

    # Read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="Path to configuration file.")
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # Choose gpu
    gpu_idx = config["gpu_idx"]
    jax.config.update('jax_default_device', jax.devices()[gpu_idx])

    # Game
    game_name = config["game_name"]

    # Logs
    now = datetime.now().strftime("%d-%m__%H:%M")
    writer_path = config["writer_path"]
    agent_name = config["agent_name"]
    if writer_path is not None:
        writer_path = os.path.join(writer_path, game_name, agent_name, now)

    # Checkpoints
    checkpoint_path = config["checkpoint_path"]
    if checkpoint_path is not None:
        checkpoint_path = os.path.join(checkpoint_path, game_name, agent_name)

    # Agent parameters
    agent_kwargs = config["agent_kwargs"]
    agent_kwargs["game_name"] = game_name
    agent_kwargs["config_kwargs"] = config["config_kwargs"]
    agent_kwargs["training_kwargs"] = config["training_kwargs"]
    agent_kwargs["pi_network_kwargs"] = config["pi_network_kwargs"]
    agent_kwargs["val_network_kwargs"] = config["val_network_kwargs"]
    agent_kwargs["pi_optimizer_kwargs"] = config["pi_optimizer_kwargs"]
    agent_kwargs["val_optimizer_kwargs"] = config["val_optimizer_kwargs"]
    agent_kwargs["writer_path"] = writer_path
    agent_kwargs["checkpoint_path"] = checkpoint_path

    agent = ProxAgent(**agent_kwargs)

    try:
        agent.fit()
    except KeyboardInterrupt:
        gc.collect()
        pass
