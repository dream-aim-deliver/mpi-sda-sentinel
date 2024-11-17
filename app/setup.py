from datetime import datetime
from logging import Logger
import os
import re
from typing import Tuple

from app.sdk.file_repository import FileRepository
from app.sdk.kernel_plackster_gateway import KernelPlancksterGateway
from app.sdk.models import ProtocolEnum

def string_validator(value: str, arg_name: str) -> str:
    value_error_flag = False
    value_error_msg = ""

    if value == "":
        value_error_msg += f"The string must not be empty. "
        raise ValueError(value_error_msg)

    v2 = re.sub(r"[^a-zA-Z0-9_\./-]", "", value)
    if value != v2:
        value_error_flag = True
        value_error_msg += f"The string  must contain only alphanumeric characters, underscores, slashes, and dots. Other characters are not allowed. Found: '{set(value) - set(v2)}' "

    first_char = value[0]
    if first_char == "/":
        value_error_flag = True
        value_error_msg += f"The string provided must not start with a slash. "

    if value_error_flag:
        value_error_msg += f"\nThe value for '{arg_name}' provided was: '{value}'"
        raise ValueError(value_error_msg)

    return value

def datetime_parser(date_string: str) -> datetime:
    dt = datetime.strptime(date_string, "%Y-%m-%dT%H:%M")
    return dt

def _setup_kernel_planckster(
    job_id: int,
    logger: Logger,
    kernel_planckster_host: str,
    kernel_planckster_port: int,
    kernel_planckster_auth_token: str,
    kernel_planckster_scheme: str,
) -> KernelPlancksterGateway:

    try:

        logger.info(f"{job_id}: Setting up Kernel Planckster Gateway.")

        # Setup the Kernel Planckster Gateway
        kernel_planckster = KernelPlancksterGateway(
            host=kernel_planckster_host,
            port=kernel_planckster_port,
            auth_token=kernel_planckster_auth_token,
            scheme=kernel_planckster_scheme,
        )
        kernel_planckster.ping()
        logger.info(f"{job_id}: Kernel Planckster Gateway setup successfully.")

        return kernel_planckster

    except Exception as error:
        logger.error(
            f"{job_id}: Unable to setup the Kernel Planckster Gateway. Error:\n{error}"
        )
        raise error



def _setup_file_repository(
    job_id: int,
    storage_protocol: ProtocolEnum,
    logger: Logger,
) -> FileRepository:

    try:
        logger.info(f"{job_id}: Setting up the File Repository.")

        file_repository = FileRepository(
            protocol=storage_protocol,
        )

        logger.info(f"{job_id}: File Repository setup successfully.")

        return file_repository

    except Exception as error:
        logger.error(f"{job_id}: Unable to setup the File Repository. Error:\n{error}")
        raise error



def setup(
    job_id: int,
    logger: Logger,
    kp_auth_token: str,
    kp_host: str,
    kp_port: int,
    kp_scheme: str,
) -> Tuple[KernelPlancksterGateway, ProtocolEnum, FileRepository]:
    """
    Setup the Kernel Planckster Gateway, the storage protocol and the file repository.

    NOTE: needs and '.env' file within context.
    """

    try:
        kernel_planckster = _setup_kernel_planckster(
            job_id, logger, kp_host, kp_port, kp_auth_token, kp_scheme
        )

        logger.info(f"{job_id}: Checking storage protocol.")
        protocol = ProtocolEnum(os.getenv("STORAGE_PROTOCOL", ProtocolEnum.S3.value))

        if protocol not in [ProtocolEnum.S3, ProtocolEnum.LOCAL]:
            logger.error(f"{job_id}: STORAGE_PROTOCOL must be either 's3' or 'local'.")
            raise ValueError("STORAGE_PROTOCOL must be either 's3' or 'local'.")

        logger.info(f"{job_id}: Storage protocol: {protocol}")

        file_repository = _setup_file_repository(job_id, protocol, logger)

        return kernel_planckster, protocol, file_repository

    except Exception as error:
        logger.error(f"{job_id}: Unable to setup. Error:\n{error}")
        raise error
