from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

from models import PipelineRequestModel

# TODO: Add time to measure server response time

app = FastAPI()

load_dotenv()
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.captureWarnings(True)


@app.put("/pipeline")
async def process_pipeline(request_data: PipelineRequestModel):
    lfn = request_data.lfn
    # data_sources = request_data.data_sources

    # You can process the data_sources as needed here
    results = []

    # for source in data_sources:
    #     source_name = source.source
    #     query = source.q
    #     latitude = query.latitude
    #     longitude = query.longitude

        # Append results for each source to the results list
        # results.append({"source": source_name, "latitude": latitude, "longitude": longitude})

    # response_data = {"message": "Pipeline processing completed for " + name, "results": results}
    return JSONResponse(content="Pipeline processing completed for " + lfn)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
