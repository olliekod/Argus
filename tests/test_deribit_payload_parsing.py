from src.connectors.deribit_client import DeribitClient


def test_deribit_coerce_result_list_from_dict():
    payload = {"result": [{"instrument_name": "BTC-TEST"}]}
    items = DeribitClient._coerce_result_list(payload, "book_summary")
    assert isinstance(items, list)
    assert items[0]["instrument_name"] == "BTC-TEST"


def test_deribit_coerce_result_list_from_list():
    payload = [{"instrument_name": "BTC-TEST"}]
    items = DeribitClient._coerce_result_list(payload, "book_summary")
    assert isinstance(items, list)
    assert items[0]["instrument_name"] == "BTC-TEST"
