import yaml
import argparse


from experiments.experiments import ExperimentGenerator

import jax

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='Path to configuration file.')
    args = parser.parse_args()

    config_path = args.config

    exp_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    # Choose gpu
    gpu_idx = exp_config['gpu_idx']
    jax.config.update('jax_default_device', jax.devices()[gpu_idx])
    del exp_config['gpu_idx']

    exp_generator = ExperimentGenerator(**exp_config)
    exp_generator.run()
