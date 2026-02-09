"""Option chain normalization utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from src.connectors.tastytrade_rest import parse_rfc3339_nano


def _first_present(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _parse_expiry(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str) and ("T" in value or "Z" in value or "+" in value):
        try:
            return parse_rfc3339_nano(value).date().isoformat()
        except ValueError:
            return None
    if isinstance(value, str):
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_tastytrade_nested_chain(raw: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Normalize tastytrade nested option chain response into a flat list."""
    if not raw:
        return []

    data = raw.get("data", raw)
    if not isinstance(data, dict):
        return []

    underlying = _first_present(
        [
            data.get("underlying-symbol"),
            data.get("symbol"),
            data.get("underlying"),
        ]
    )

    expirations = (
        data.get("expirations")
        or data.get("items")
        or data.get("option-chains")
        or []
    )

    normalized: list[Dict[str, Any]] = []

    for expiration in expirations:
        expiry_raw = _first_present(
            [
                expiration.get("expiration-date"),
                expiration.get("expiration"),
                expiration.get("expiration-date-time"),
                expiration.get("date"),
            ]
        )
        expiry = _parse_expiry(expiry_raw)

        strikes = (
            expiration.get("strikes")
            or expiration.get("strike-prices")
            or expiration.get("strike-price-list")
            or []
        )

        for strike in strikes:
            if isinstance(strike, dict):
                strike_price = _first_present(
                    [
                        strike.get("strike-price"),
                        strike.get("strike"),
                        strike.get("price"),
                        strike.get("strike_price"),
                    ]
                )
            else:
                strike_price = strike

            strike_value = _float_or_none(strike_price)

            for right_label, option_key in (("C", "call"), ("P", "put")):
                option_data = strike.get(option_key) if isinstance(strike, dict) else None
                if not option_data:
                    continue

                option_symbol = _first_present(
                    [
                        option_data.get("streamer-symbol"),
                        option_data.get("symbol"),
                        option_data.get("occ-symbol"),
                    ]
                )
                multiplier = _int_or_default(
                    _first_present(
                        [
                            option_data.get("multiplier"),
                            option_data.get("contract-size"),
                        ]
                    ),
                    100,
                )

                currency = _first_present(
                    [
                        option_data.get("currency"),
                        expiration.get("currency"),
                        data.get("currency"),
                    ]
                ) or "USD"

                exchange = _first_present(
                    [
                        option_data.get("exchange"),
                        option_data.get("listing-exchange"),
                        expiration.get("exchange"),
                    ]
                )

                meta = {
                    "root": _first_present(
                        [
                            option_data.get("root-symbol"),
                            option_data.get("root"),
                            data.get("root-symbol"),
                        ]
                    ),
                    "product_type": _first_present(
                        [
                            option_data.get("product-type"),
                            expiration.get("product-type"),
                            data.get("product-type"),
                        ]
                    ),
                    "settlement_type": _first_present(
                        [
                            option_data.get("settlement-type"),
                            expiration.get("settlement-type"),
                        ]
                    ),
                    "expiration_type": _first_present(
                        [
                            option_data.get("expiration-type"),
                            expiration.get("expiration-type"),
                        ]
                    ),
                }

                meta = {key: value for key, value in meta.items() if value is not None}

                normalized.append(
                    {
                        "provider": "tastytrade",
                        "underlying": underlying,
                        "option_symbol": option_symbol,
                        "expiry": expiry,
                        "right": right_label,
                        "strike": strike_value,
                        "multiplier": multiplier,
                        "currency": currency,
                        "exchange": exchange,
                        "meta": meta,
                    }
                )

    normalized.sort(
        key=lambda item: (
            item.get("expiry") or "",
            item.get("strike") if item.get("strike") is not None else -1.0,
            item.get("right") or "",
        )
    )
    return normalized
