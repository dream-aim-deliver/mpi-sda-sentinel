from logging import Logger
import sys
from sentinelhub import SHConfig

def get_scraping_config(
        job_id: int,
        logger: Logger,
        sentinel_client_id: str,
        sentinel_client_secret: str,
        ):

    try:
        logger.info(f"{job_id}: Setting up Sentinel client.")

        if not all([sentinel_client_id, sentinel_client_secret]):
            logger.error(f"{job_id}:CLIENT_ID or CLIENT_SECRET not found in environment variables.")
            raise ValueError("Missing required API credentials.")
        
        config = SHConfig()
        config.sh_client_id = sentinel_client_id
        config.sh_client_secret = sentinel_client_secret
        config.save()

        return config

    except Exception as error:
        logger.error(f"{job_id}: Unable to setup the Sentinel client. Error:\n{error}")
        sys.exit(1)