# server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
from models import PipelineRequestModel, QueryModel
from sentinel import SentinelHubPipelineElement
import time

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)


@app.put("/pipeline")
async def process_pipeline(request_data: PipelineRequestModel):
    """
    Process the pipeline request using Sentinel Hub Pipeline.

    Args:
        request_data (PipelineRequestModel): A model representing the pipeline request data.

    Returns:
        JSONResponse: A response object with the result of the processing.
    """
    start_time = time.time()  # Record start time for response time measurement
    try:
        # Create an instance of SentinelHubPipelineElement with the request data
        sentinel_pipeline = SentinelHubPipelineElement(request_data, request_data.start_date, request_data.end_date)

        # Execute the pipeline process
        sentinel_pipeline.execute()

        # Calculate response time
        response_time = time.time() - start_time
        response_data = {
            "message": f"Pipeline processing completed for {request_data.lfn}",
            "response_time": f"{response_time:.2f} seconds"
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        logging.error(f"Error in processing pipeline: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred.")


# Running the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
