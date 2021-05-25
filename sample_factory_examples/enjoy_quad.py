import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory_examples.quad_runner import register_custom_components, custom_parse_args


def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    status = enjoy(cfg)
    return status

# enjoy_quad
if __name__ == '__main__':
    sys.exit(main())
