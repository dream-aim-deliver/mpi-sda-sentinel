#!/usr/bin/env bash

source .env

# Check that SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET are set
if [ -z "$SENTINEL_CLIENT_ID" ] || [ -z "$SENTINEL_CLIENT_SECRET" ]; then
  echo "Please set the SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET environment variables."
  exit 1
fi


python sentinel_scraper.py \
 --case-study-name="disaster" \
 --tracer-id="potato" \
 --job-id="2" \
 --start_date=2024-09-15T09:00 \
 --end_date=2024-09-15T12:00 \
 --interval=60 \
 --long_left=-156.708984 \
 --lat_up=20.759645 \
 --long_right=-156.299744 \
 --lat_down=20.955027 \
 --log-level="INFO" \
 --kp_auth_token test123 --kp_host localhost --kp_port 8000 --kp_scheme http \
 --sentinel_client_id "${SENTINEL_CLIENT_ID}"  --sentinel_client_secret "${SENTINEL_CLIENT_SECRET}" \
 --evalscript_bands_path "https://gist.githubusercontent.com/Rahul-7131/b02d5614401ba654904ff509039def15/raw/3867e78b12bf7d7dff44810c548ed20797b367ea/wildfire_evalscript.js" \
 --evalscript_truecolor_path "https://gist.githubusercontent.com/Rahul-7131/b02d5614401ba654904ff509039def15/raw/3867e78b12bf7d7dff44810c548ed20797b367ea/truecolor_wildfire.js" \