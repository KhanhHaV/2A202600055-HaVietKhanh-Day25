from __future__ import annotations

import time
from dataclasses import dataclass

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker, CircuitOpenError
from reliability_lab.providers import FakeLLMProvider, ProviderError, ProviderResponse


@dataclass(slots=True)
class GatewayResponse:
    text: str
    route: str
    provider: str | None
    cache_hit: bool
    latency_ms: float
    estimated_cost: float
    error: str | None = None


class ReliabilityGateway:
    """Routes requests through cache, circuit breakers, and fallback providers."""

    def __init__(
        self,
        providers: list[FakeLLMProvider],
        breakers: dict[str, CircuitBreaker],
        cache: ResponseCache | SharedRedisCache | None = None,
    ):
        self.providers = providers
        self.breakers = breakers
        self.cache = cache

    def complete(self, prompt: str) -> GatewayResponse:
        """Return a reliable response or a static fallback."""
        start_time = time.monotonic()
        if self.cache is not None:
            try:
                cached, score = self.cache.get(prompt)
                if cached is not None:
                    latency = (time.monotonic() - start_time) * 1000
                    return GatewayResponse(cached, f"cache_hit:{score:.2f}", None, True, latency, 0.0)
            except Exception as e:
                # Cache failed (e.g. Redis down), degrade gracefully
                pass

        last_error: str | None = None
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            try:
                response: ProviderResponse = breaker.call(provider.complete, prompt)
                if self.cache is not None:
                    try:
                        self.cache.set(prompt, response.text, {"provider": provider.name})
                    except Exception:
                        pass
                
                route_type = "primary" if provider == self.providers[0] else "fallback"
                route = f"{route_type}:{provider.name}"
                
                # End-to-end latency includes routing/cache overhead
                total_latency = (time.monotonic() - start_time) * 1000
                
                return GatewayResponse(
                    text=response.text,
                    route=route,
                    provider=provider.name,
                    cache_hit=False,
                    latency_ms=total_latency,
                    estimated_cost=response.estimated_cost,
                )
            except (ProviderError, CircuitOpenError) as exc:
                last_error = str(exc)
                continue

        total_latency = (time.monotonic() - start_time) * 1000
        return GatewayResponse(
            text="The service is temporarily degraded. Please try again soon.",
            route="static_fallback",
            provider=None,
            cache_hit=False,
            latency_ms=total_latency,
            estimated_cost=0.0,
            error=last_error,
        )
