"""Entry point for the refactored SFT training workflow."""

from sft.config import parse_args


def main() -> None:
    config = parse_args()
    from sft.trainer import train

    train(config)


if __name__ == "__main__":
    main()
