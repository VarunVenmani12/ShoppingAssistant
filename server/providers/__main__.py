"""Print the configured providers and their status: python -m server.providers"""

import json

from server.providers.registry import build_providers, load_provider_env, parse_provider_config


def main() -> None:
    env = load_provider_env()
    config = parse_provider_config(env)
    print(f"SHOPPING_PROVIDERS -> enabled: {', '.join(config.enabled)}")
    if config.unknown:
        print(f"Ignored unknown provider(s): {', '.join(config.unknown)}")
    for provider in build_providers(env):
        print(json.dumps(provider.status().to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
