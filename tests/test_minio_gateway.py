import os
import tempfile

from app.sdk.minio_gateway import MinIORepository
from app.sdk.models import LFN, Protocol, DataSource


def test_minio_gateway(minio_config):
    minio = MinIORepository(
        host=minio_config["host"],
        port=minio_config["port"],
        access_key=minio_config["access_key"],
        secret_key=minio_config["secret_key"],
    )
    assert minio.user == minio_config["access_key"]


def test_minio_gateway_get_client(minio):
    client = minio.get_client()
    assert client is not None


def test_minio_gateway_create_bucket_if_not_exists(minio):
    bucket_name = "test-bucket"
    minio.create_bucket_if_not_exists(bucket_name)
    assert bucket_name in minio.list_buckets()


def test_minio_gateway_list_objects(minio: MinIORepository):
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"test")
        f.seek(0)
        bucket_name = "test-bucket"
        object_name = "test-object"
        minio._upload_file(bucket_name, object_name, f.name)
        objects = minio.list_objects(bucket_name)
        assert object_name in objects


def test_upload_download_file(minio: MinIORepository):
    lfn = LFN(
        protocol=Protocol.S3,
        tracer_id="test",
        job_id=1,
        source=DataSource.SENTINEL,
        #relative_path="root/data2_climate-sdamarker.txt",
    )

    minio.create_bucket_if_not_exists("default")

    content = b"test"
    with tempfile.NamedTemporaryFile() as f:
        f.write(content)
        f.seek(0)
        minio.upload_file(lfn, f.name)

    f = tempfile.NamedTemporaryFile()
    minio.download_file(lfn, f.name)
    with open(f.name, "rb") as file:
        data = file.read()
        assert data == content
    # delete file
    os.remove(f.name)


def test_lfn_to_pfn(minio: MinIORepository):
    lfn = LFN(
        protocol=Protocol.S3,
        tracer_id="test",
        job_id=1,
        source=DataSource.SENTINEL,
        #relative_path="root/data2_climate.csv",
    )
    pfn = minio.lfn_to_pfn(lfn)
    assert "root/data2_climate" in pfn
    assert "-sdamarker" in pfn
    assert "s3://localhost:9000/default/test/sentinel/1/" in pfn
    augmented_lfn = LFN(
        protocol=Protocol.S3,
        tracer_id="test",
        job_id=1,
        source=DataSource.SENTINEL,
        #relative_path="root/data2_climate-sdamarker.csv",
    )

    pfn2 = minio.lfn_to_pfn(augmented_lfn)
    assert (
        pfn2
        == f"s3://localhost:9000/default/test/sentinel/1/root/data2_climate-sdamarker.csv"
    )


def test_pfn_to_lfn(minio: MinIORepository):
    pfn = "s3://localhost:9000/default/test/sentinel/1/root/data2_climate-sdamarker.csv"
    lfn = minio.pfn_to_lfn(pfn)
    assert lfn.protocol == Protocol.S3
    assert lfn.tracer_id == "test"
    assert lfn.job_id == 1
    assert lfn.source == DataSource.SENTINEL
    #assert lfn.relative_path == "root/data2_climate-sdamarker.csv"


def test_pfn_to_object_name(minio: MinIORepository):
    pfn = "s3://localhost:9000/default/test/sentinel/1/root/data2_climate-sdamarker.csv"
    object_name = minio.pfn_to_object_name(pfn)
    assert object_name == "test/sentinel/1/root/data2_climate-sdamarker.csv"


def test_object_name_to_pfn(minio: MinIORepository):
    object_name = "test/sentinel/1/root/data2_climate-sdamarker.csv"
    pfn = minio.object_name_to_pfn(object_name)
    assert (
        pfn
        == "s3://localhost:9000/default/test/sentinel/1/root/data2_climate-sdamarker.csv"
    )
