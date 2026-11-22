import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from config import TestConfig  # noqa: E402
from database.models import User  # noqa: E402
from extensions import db  # noqa: E402

SAMPLES = ROOT / "data" / "sample_uploads"
XLSX_SAMPLE = SAMPLES / "ibm_hi_small_2022-09-10_1400-1415.xlsx"
CSV_SAMPLE = SAMPLES / "ibm_hi_small_2022-09-09_1000-1030.csv"
STAGE1 = ROOT.parent / "CoDetect_Financial_Fraud_Detection"
PASSWORD = "Passw0rd!x"


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("codetect")

    class Cfg(TestConfig):
        UPLOAD_FOLDER = tmp / "uploads"
        REPORTS_FOLDER = tmp / "reports"

    application = create_app(Cfg)
    with application.app_context():
        db.create_all()
    return application


@pytest.fixture(autouse=True)
def fresh_db(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        for name, email, role in (("Ada Admin", "admin@example.com", "Admin"),
                                  ("Ravi Analyst", "analyst@example.com", "User")):
            u = User(name=name, email=email, user_type=role)
            u.set_password(PASSWORD)
            db.session.add(u)
        db.session.commit()
    yield
    with app.app_context():
        db.session.remove()


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, email):
    return client.post("/login", data={"email": email, "password": PASSWORD}, follow_redirects=False)


@pytest.fixture
def admin(app):
    c = app.test_client()          # each role gets its own client / session
    login(c, "admin@example.com")
    return c


@pytest.fixture
def analyst(app):
    c = app.test_client()
    login(c, "analyst@example.com")
    return c


def upload(client, path=XLSX_SAMPLE, name="test window", content=None, filename=None):
    data = content if content is not None else path.read_bytes()
    import io
    return client.post("/admin/datasets/upload", data={
        "dataset_name": name, "source_name": "IBM AML HI-Small",
        "file": (io.BytesIO(data), filename or path.name)}, content_type="multipart/form-data",
        follow_redirects=True)


@pytest.fixture
def uploaded(admin, app):
    r = upload(admin)
    assert r.status_code == 200
    from database.models import FinancialDataset
    with app.app_context():
        return db.session.query(FinancialDataset).one().id


@pytest.fixture
def analyzed(admin, uploaded):
    r = admin.post(f"/admin/datasets/{uploaded}/detect", follow_redirects=True)
    assert r.status_code == 200
    return uploaded


@pytest.fixture
def no_models_app(tmp_path):
    """App whose ML_ARTIFACTS_DIR is empty -> graceful 'models unavailable' handling."""
    class Cfg(TestConfig):
        ML_ARTIFACTS_DIR = tmp_path / "empty_artifacts"
        UPLOAD_FOLDER = tmp_path / "uploads"
        REPORTS_FOLDER = tmp_path / "reports"
    Cfg.ML_ARTIFACTS_DIR.mkdir()
    a = create_app(Cfg)
    with a.app_context():
        db.create_all()
        u = User(name="Ada Admin", email="admin@example.com", user_type="Admin")
        u.set_password(PASSWORD)
        db.session.add(u)
        db.session.commit()
    return a
