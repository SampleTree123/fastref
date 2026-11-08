"""Read named inference presets. Owner: configs/inference/README.md.

Explicit CLI arguments override preset values; without a preset the entrypoint's
existing defaults (including fixed-case settings) are preserved.
"""
import argparse
import json
from pathlib import Path

from project_layout import resolve_path


def parse_with_inference_settings(parser, argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--inference-config', type=Path)
    requested, _ = probe.parse_known_args(argv)
    parser.add_argument('--inference-config', type=Path,
                        help='Named generation settings; explicit CLI options take precedence')
    if requested.inference_config is not None:
        try:
            payload = json.loads(resolve_path(requested.inference_config).read_text())
            if payload.get('schema') != 'datamake_inference_settings_v1':
                raise ValueError('Unknown inference settings schema')
            settings = payload['settings']
            allowed = {'steps', 'seed', 'cfg_scale', 'stg_scale', 'guidance_rescale',
                       'stg_blocks', 'negative_prompt', 'guidance_batch_size'}
            if not isinstance(settings, dict) or set(settings) - {'ref_cfg'} != allowed:
                raise ValueError('Inference settings must contain exactly: ' + ', '.join(sorted(allowed)))
            if type(settings['steps']) is not int or settings['steps'] < 2:
                raise ValueError('steps must be an integer >= 2')
            if type(settings['seed']) is not int or settings['seed'] < 0:
                raise ValueError('seed must be a nonnegative integer')
            # Uses the same validation and default-negative-prompt behavior as workers.
            from run_fixed_case_inference import guidance_contract
            guidance_contract(settings['cfg_scale'], settings['stg_scale'],
                              settings['guidance_rescale'], settings['stg_blocks'],
                              settings['negative_prompt'], settings['guidance_batch_size'], settings.get('ref_cfg', 1.0))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            parser.error('Invalid inference config: ' + str(error))
        parser.set_defaults(**settings)
    return parser.parse_args(argv)
