import os
from dotenv import load_dotenv
import pytest

from app.sdk.minio_gateway import MinIORepository 


load_dotenv()


@pytest.fixture
def minio_config():
    return {
        "host": os.getenv("MINIO_HOST","localhost"),
        "port": os.getenv("MINIO_PORT","9000"),
        "access_key": os.getenv("MINIO_ACCESS_KEY","minioadmin"),
        "secret_key": os.getenv("MINIO_SECRET_KEY","minioadmin"),
    }

@pytest.fixture
def minio(minio_config):
    minio = MinIORepository(
        host=minio_config["host"],
        port=minio_config["port"],
        access_key=minio_config["access_key"],
        secret_key=minio_config["secret_key"],
    )
    return minio