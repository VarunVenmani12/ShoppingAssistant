"""
Amazon India and Flipkart provider placeholders.

These implement the provider interface but deliberately make NO network
calls and return NO data. Live integration is planned for a later phase:

* Amazon: the Creators API (PA-API 5.0 was retired in May 2026). Access needs
  an approved Amazon.in Associates account with recent qualifying sales.
* Flipkart: the Affiliate API (Affiliate ID + API token).

Every data operation raises ProviderNotConfiguredError with a clear message,
so the assistant reports "not configured" instead of fabricating results.
Credential values are only checked for presence; they are never logged or
returned.
"""

from __future__ import annotations

from collections.abc import Mapping

from server.product_store import SearchFilters
from server.providers.base import (
    Offer,
    Product,
    ProductProvider,
    ProviderNotConfiguredError,
    ProviderStatus,
)


class MarketplacePlaceholder(ProductProvider):
    display_name = "Marketplace"
    credential_keys: tuple[str, ...] = ()

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        env = env or {}
        self.credentials_present = bool(self.credential_keys) and all(
            env.get(key, "").strip() for key in self.credential_keys
        )

    def status(self) -> ProviderStatus:
        if self.credentials_present:
            message = (
                f"{self.display_name} credentials were found, but live integration is not "
                "implemented yet. No marketplace data is available."
            )
        else:
            message = (
                f"{self.display_name} live integration is not configured "
                f"(needs {', '.join(self.credential_keys)}). No marketplace data is available."
            )
        return ProviderStatus(
            self.name, available=False, live=False, message=message,
            credentials_present=self.credentials_present,
        )

    def _not_configured(self) -> ProviderNotConfiguredError:
        return ProviderNotConfiguredError(self.status().message)

    def search(self, filters: SearchFilters) -> list[Product]:
        raise self._not_configured()

    def get_product(self, product_id: str) -> Product | None:
        raise self._not_configured()

    def get_offers(self, product_id: str) -> list[Offer]:
        raise self._not_configured()


class AmazonProvider(MarketplacePlaceholder):
    name = "amazon"
    display_name = "Amazon India"
    credential_keys = ("AMAZON_CREATORS_CLIENT_ID", "AMAZON_CREATORS_CLIENT_SECRET", "AMAZON_PARTNER_TAG")


class FlipkartProvider(MarketplacePlaceholder):
    name = "flipkart"
    display_name = "Flipkart"
    credential_keys = ("FLIPKART_AFFILIATE_ID", "FLIPKART_AFFILIATE_TOKEN")
