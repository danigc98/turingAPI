## FastAPI Image Detection Project

This is a FastAPI project that provides an endpoint for image detection using a YOLOv4 model. It accepts image uploads and returns bounding boxes for detected objects. 

Before running the project, ensure you have the necessary dependencies installed. You can install them using pip:

### Prerequisites 
``` bash
pip install -r requirements.txt
```
### Running the Project

To run the project, use the following command:
``` bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
``` 
This will start the FastAPI application and make it accessible at http://localhost:8000.
You can explore the API and its functionality using FastAPI's built-in Swagger UI. Simply visit http://localhost:8000/docs in your browser to access the API documentation.

### Usage
POST /detect

This endpoint allows you to upload an image and receive bounding boxes for detected objects. You can also specify a detection threshold (default is 0.25).

    Request:
        Upload an image file (jpg or png format).
        Optionally, provide the threshold query parameter (0.0 to 1.0) to adjust the detection threshold.

    Response:
        The response contains the number of detections and a list of detected objects with their bounding box coordinates and labels.

Example: 

``` bash
curl -X 'POST' \
  'http://0.0.0.0/detect' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@yourImagePath.jpeg;type=image/jpeg'
  
```

### Docker

You can also run the project using Docker. To do so, first build the Docker image:

``` bash
docker build -t fastapi-image-detection .
```

Then run the container:

``` bash
docker run -p 8000:8000 fastapi-image-detection
```



