import logging
import sys
import json
from app.config import SUPPORTED_DATASET_EVALSCRIPTS
from app.scraper import scrape
from app.sdk.scraped_data_repository import ScrapedDataRepository
from app.setup import setup, string_validator


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
    datasets_evalscripts: dict[str,list[str]],
    resolution: int,
    sentinel_client_id: str,
    sentinel_client_secret: str,
    kp_host: str,
    kp_port: int,
    kp_auth_token: str,
    kp_scheme: str,
    log_level: str = "WARNING",
) -> None:

    try:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    
        if not all([case_study_name, job_id, tracer_id, long_left, lat_down, long_right, lat_up, start_date, end_date]):
            raise ValueError(f"case_study_name, job_id, tracer_id, coordinates, and date range must all be set.")

        string_variables = {
            "case_study_name": case_study_name,
            "job_id": job_id,
            "tracer_id": tracer_id,
        }

        logger.info(f"Validating string variables:  {string_variables}")

        for name, value in string_variables.items():
            string_validator(f"{value}", name)

        logger.info(f"String variables validated successfully!")
        
        final_datasaet_evalscripts = {}
        dataset_names = datasets_evalscripts.keys()
        for dataset_name in dataset_names:
            if dataset_name not in SUPPORTED_DATASET_EVALSCRIPTS.keys():
                logger.error(
                    f"Dataset {dataset_name} not supported. Use one of {SUPPORTED_DATASET_EVALSCRIPTS.keys()}"
                )
                sys.exit(1)
            requested_evalscripts = datasets_evalscripts[dataset_name]
            supported_evalscripts = [x['name'] for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]["supported_evalscripts"]]
            for evalscript in requested_evalscripts:
                if evalscript not in supported_evalscripts:
                    logger.error(
                        f"Evalscript {evalscript} not supported. Use one of {SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]['supported_evalscripts']}"
                    )
                    raise ValueError(
                        f"Evalscript {evalscript} not supported for {dataset_name}. Use one of {SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]['supported_evalscripts']}"
                    )
            final_datasaet_evalscripts[dataset_name] = SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]
            final_datasaet_evalscripts[dataset_name]["evalscripts"] = [x for x in SUPPORTED_DATASET_EVALSCRIPTS[dataset_name]["supported_evalscripts"] if x["name"] in requested_evalscripts]
            
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
        dataset_evalscripts=final_datasaet_evalscripts,
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
        "--datasets-evalscripts",
        type=json.loads,
        required=True,
        help="dictionary in the format {\"dataset_name\": [evalscript_path1, evalscript_path2, ...]}",
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
        required=True,
        help="client id",
    )

    parser.add_argument(
        "--sentinel_client_secret",
        type=str,
        required=True,
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
        datasets_evalscripts=args.datasets_evalscripts,
        resolution=args.resolution,
        sentinel_client_id=args.sentinel_client_id,
        sentinel_client_secret=args.sentinel_client_secret,
        kp_host=args.kp_host,
        kp_port=args.kp_port,
        kp_auth_token=args.kp_auth_token,
        kp_scheme=args.kp_scheme
    )


