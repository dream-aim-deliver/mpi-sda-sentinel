import logging
import sys
from app.scraper import scrape
from app.sdk.scraped_data_repository import ScrapedDataRepository
from app.setup import datetime_parser,setup, string_validator


from app.setup_scraping_client import get_scraping_config


def main(
    case_study_name: str,
    job_id: int,
    tracer_id: str,
    long_left: float,
    lat_down: float,
    long_right: float,
    lat_up: float,
    start_date: str,
    end_date: str,
    interval: int,
    image_dir: str,
    dataset_name:str,
    resolution: int,
    evalscript_bands_path:str,
    evalscript_truecolor_path:str,
    sentinel_client_id: str,
    sentinel_client_secret: str,
    kp_host: str,
    kp_port: str,
    kp_auth_token: str,
    kp_scheme: str,
    log_level: str = "WARNING",
) -> None:

    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)

    
        if not all([case_study_name, job_id, tracer_id, long_left, lat_down, long_right, lat_up, start_date, end_date]):
            raise ValueError(f"case_study_name, job_id, tracer_id, coordinates, and date range must all be set.")

        string_variables = {
            "case_study_name": case_study_name,
            "job_id": job_id,
            "tracer_id": tracer_id,
            "dataset_name": dataset_name,
        }

        logger.info(f"Validating string variables:  {string_variables}")

        for name, value in string_variables.items():
            string_validator(f"{value}", name)

        logger.info(f"String variables validated successfully!")
        

        logger.info(f"Setting up scraper for case study: {case_study_name}")

        kernel_planckster, protocol, file_repository = setup(
            job_id=job_id,
            logger=logger,
            kp_auth_token=kp_auth_token,
            kp_host=kp_host,
            kp_port=kp_port,
            kp_scheme=kp_scheme,
        )

        scraped_data_repository = ScrapedDataRepository(
            protocol=protocol,
            kernel_planckster=kernel_planckster,
            file_repository=file_repository,
        )

        sentinel_config = get_scraping_config(
            job_id=job_id,
            logger=logger,
            sentinel_client_id=sentinel_client_id,
            sentinel_client_secret=sentinel_client_secret
        )

    except Exception as error:
        logger.error(f"Unable to setup the scraper. Error: {error}")
        sys.exit(1)


    job_output = scrape(
        case_study_name=case_study_name,
        job_id=job_id,
        tracer_id=tracer_id,
        scraped_data_repository=scraped_data_repository,
        log_level=log_level,
        long_left=long_left,
        lat_down=lat_down,
        long_right=long_right,
        lat_up=lat_up,
        sentinel_config=sentinel_config,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        image_dir=image_dir,
        evalscript_bands_path=evalscript_bands_path,
        evalscript_truecolor_path=evalscript_truecolor_path,
        dataset_name=dataset_name,
        resolution=resolution
    )

    logger.info(f"{job_id}: Scraper finished with state: {job_output.job_state.value}")

    if job_output.job_state.value == "failed":
        sys.exit(1)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Scrape data from Sentinel datacollection.")

    parser.add_argument(
        "--case-study-name",
        type=str,
        help="The name of the case study",
        required=True,
    )

    parser.add_argument(
        "--job-id",
        type=str,
        help="The job id",
        required=True,
    )

    parser.add_argument(
        "--tracer-id",
        type=str,
        help="The tracer id",
        required=True,
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="The log level to use when running the scraper. Possible values are DEBUG, INFO, WARNING, ERROR, CRITICAL. Set to WARNING by default.",
    )

    parser.add_argument(
        "--long_left",
        type=float,
        default="0",
        help="leftmost longtude ~ left edge of bbox ",
    )

    parser.add_argument(
        "--lat_down",
        type=float,
        default="0",
        help="bottommost lattitude ~ bottom edge of bbox ",
    )

    parser.add_argument(
        "--long_right",
        type=float,
        default="0.1",
        help="rightmost longtude ~ right edge of bbox ",
    )

    parser.add_argument(
        "--lat_up",
        type=float,
        default="0.1",
        help="topmost lattitude ~ top edge of bbox ",
    
    )

    parser.add_argument(
        "--start_date",
        type=str,
        help="start date",
        required=True,
    )

    parser.add_argument(
        "--end_date",
        type=str,
        help="end date",
        required=True,
    )

    parser.add_argument(
        "--interval",
        type=int,
        help="Time interval between scraping events, in minutes.",
        required=True,
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./.tmp",
        help="image dir",
    )

    parser.add_argument(
        "--evalscript_bands_path", 
        type=str, 
        required=True, 
        help="Path to Evalscript for Bands Configuration"
    )

    parser.add_argument(
        "--evalscript_truecolor_path",
        type=str,
        required=False,
        help="Path to truecolor Evalscript file for augmentation"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SENTINEL5P",
        help="dataset configuration",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default=60,
        help="resolution",
    )

    parser.add_argument(
        "--sentinel_client_id",
        type=str,
        default="60",
        help="client id",
    )

    parser.add_argument(
        "--sentinel_client_secret",
        type=str,
        default="60",
        help="client secret ",
    )

    parser.add_argument(
        "--kp_host",
        type=str,
        help="kp host",
        required=True,
    )

    parser.add_argument(
        "--kp_port",
        type=int,
        help="kp port",
        required=True,
    )

    parser.add_argument(
        "--kp_auth_token",
        type=str,
        help="kp auth token",
        required=True,
        )

    parser.add_argument(
        "--kp_scheme",
        type=str,
        help="kp scheme",
        required=True,
        )


    args = parser.parse_args()


    main(
        case_study_name=args.case_study_name,
        job_id=args.job_id,
        tracer_id=args.tracer_id,
        log_level=args.log_level,
        long_left=args.long_left,
        lat_down=args.lat_down,
        long_right=args.long_right,
        lat_up=args.lat_up,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        image_dir=args.image_dir,
        dataset_name=args.dataset_name,
        resolution=args.resolution,
        evalscript_bands_path=args.evalscript_bands_path,
        evalscript_truecolor_path=args.evalscript_truecolor_path,
        sentinel_client_id=args.sentinel_client_id,
        sentinel_client_secret=args.sentinel_client_secret,
        kp_host=args.kp_host,
        kp_port=args.kp_port,
        kp_auth_token=args.kp_auth_token,
        kp_scheme=args.kp_scheme
    )


