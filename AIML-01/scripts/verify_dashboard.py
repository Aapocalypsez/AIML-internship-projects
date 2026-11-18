"""Exercise real analyst controls on reserved integration-demo records."""

import json

from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.dashboard.db import query


def widget(items, label):
    return next(item for item in items if item.label == label)


def main():
    app = AppTest.from_file("src/dashboard/app.py").run(timeout=60)
    assert not app.exception
    widget(app.selectbox, "Show cases").set_value("all")
    app.run(timeout=60)
    selected = widget(app.selectbox, "Open transaction").value
    record = query(
        "SELECT transaction_data FROM app.transactions WHERE transaction_id=%s", (selected,)
    )[0]
    if "demo_run" not in record["transaction_data"]:
        raise ValueError("Select a reserved integration-demo record before automated label testing")
    widget(app.selectbox, "Case status").set_value("reviewing")
    app.text_area[0].set_value("Automated demo workflow verification")
    widget(app.button, "Save case").click()
    app.run(timeout=60)
    assert (
        query("SELECT status FROM app.cases WHERE transaction_id=%s", (selected,))[0]["status"]
        == "reviewing"
    )
    widget(app.button, "Confirm legitimate").click()
    app.run(timeout=60)
    assert not app.exception
    assert (
        query("SELECT label FROM app.labels WHERE transaction_id=%s", (selected,))[0]["label"] == 0
    )
    result = {
        "rendered": True,
        "case_status_saved": True,
        "label_button_persisted": True,
        "transaction_id": selected,
    }
    path = load_config().data_processed_dir / "integration" / "dashboard.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
