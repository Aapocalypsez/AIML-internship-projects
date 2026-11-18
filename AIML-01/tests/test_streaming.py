from types import SimpleNamespace

from src.streaming import producer


def test_single_row_replay_with_missing_device_and_email(tmp_path, monkeypatch):
    (tmp_path / "train_transaction.csv").write_text(
        "TransactionID,TransactionDT,TransactionAmt,card1,addr1,D1,DeviceType,P_emaildomain,isFraud\n"
        "1,86400,25,100,200,1,,,0\n"
    )
    monkeypatch.setattr(
        producer, "load_config", lambda: SimpleNamespace(data_raw_dir=tmp_path)
    )

    rows = producer.load_transactions(limit=1)

    assert len(rows) == 1
    assert rows[0]["TransactionID"] == 1
    assert rows[0]["DeviceType"] is None
    assert rows[0]["P_emaildomain"] is None
    assert rows[0]["pseudo_user_id"]
    assert "isFraud" not in rows[0]
