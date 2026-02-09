from src.core.options_normalize import normalize_tastytrade_nested_chain


def test_normalize_tastytrade_nested_chain_ordering_and_fields():
    raw = {
        "data": {
            "underlying-symbol": "IBIT",
            "currency": "USD",
            "expirations": [
                {
                    "expiration-date": "2024-01-19",
                    "strikes": [
                        {
                            "strike-price": "450",
                            "call": {
                                "streamer-symbol": ".IBIT240119C450",
                                "multiplier": 100,
                                "product-type": "Equity Option",
                            },
                            "put": {
                                "streamer-symbol": ".IBIT240119P450",
                                "multiplier": 100,
                                "product-type": "Equity Option",
                            },
                        },
                        {
                            "strike-price": "440",
                            "call": {
                                "streamer-symbol": ".IBIT240119C440",
                                "multiplier": 100,
                                "product-type": "Equity Option",
                            },
                        },
                    ],
                },
                {
                    "expiration-date": "2024-01-12T00:00:00.000000000Z",
                    "strikes": [
                        {
                            "strike-price": "430",
                            "call": {"streamer-symbol": ".IBIT240112C430"},
                            "put": {"streamer-symbol": ".IBIT240112P430"},
                        }
                    ],
                },
            ],
        }
    }

    normalized = normalize_tastytrade_nested_chain(raw)

    assert len(normalized) == 5
    assert normalized[0]["expiry"] == "2024-01-12"
    assert normalized[-1]["expiry"] == "2024-01-19"
    assert normalized[0]["provider"] == "tastytrade"
    assert normalized[0]["underlying"] == "IBIT"
    assert normalized[0]["right"] in {"C", "P"}
    assert isinstance(normalized[0]["strike"], float)
    assert isinstance(normalized[0]["multiplier"], int)
    assert all(item["currency"] == "USD" for item in normalized)

    expected_order = sorted(
        normalized,
        key=lambda item: (
            item.get("expiry") or "",
            item.get("strike") if item.get("strike") is not None else -1.0,
            item.get("right") or "",
        ),
    )
    assert normalized == expected_order
