from logging import Logger
import os
from sentinelhub import SHConfig

def get_scraping_config(
        job_id: int,
        logger: Logger
        ):

    try:
        logger.info(f"{job_id}: Setting up Sentinel client.")

        """Loads SentinelHub API keyS from environment variables and validates their presence."""
        sentinel_client_id = os.getenv("CLIENT_ID")
        sentinel_client_id_secret = os.getenv("CLIENT_SECRET")
    
        if not all([sentinel_client_id, sentinel_client_id_secret]):
            logger.error(f"{job_id}:CLIENT_ID or CLIENT_SECRET not found in environment variables.")
            raise ValueError("Missing required API credentials.")
        
        config = SHConfig()
        config.sh_client_id = sentinel_client_id
        config.sh_client_secret = sentinel_client_id_secret
        config.save()

        return config

    except Exception as error:
        logger.error(f"{job_id}: Unable to setup the Twitter client. Error:\n{error}")
        raise error