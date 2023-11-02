from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import onnxruntime as rt
import cv2
import numpy as np
from src.yolo_utils import image_preprocess, postprocess, suppress_classes_except, bboxes_to_json_list
import src.config as config

# load the model
model = rt.InferenceSession("yolov4.onnx")

# start the app
app = FastAPI()


# Custom exception handler for 404
@app.exception_handler(404)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(content={"error": "Endpoint not found. Use the /detect POST endpoint for image detection."
                                          "You may visit /docs for more information."}, status_code=exc.status_code)


@app.post("/detect")
async def detect(image: UploadFile = File(...),
                 threshold: float = Query(default=0.25, description="Detection threshold (0.0 to 1.0)")):
    try:
        # Access image as a numpy array
        request_object_content = await image.read()
        image_array = np.fromstring(request_object_content, np.uint8)
        try:
            img_mat = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Unable to decode image, please ensure it is a valid image file")

        if img_mat is None:
            raise HTTPException(status_code=400, detail="Invalid image file, please use jpg or png")

        # Preprocess image
        preprocessed_img = image_preprocess(img_mat, (config.INPUT_SIZE, config.INPUT_SIZE))
        # Add batch dimension
        preprocessed_img = preprocessed_img[np.newaxis, ...].astype(np.float32)
        # Run inference
        outputs = model.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))
        input_name = model.get_inputs()[0].name
        detections = model.run(output_names, {input_name: preprocessed_img})

        # Postprocess detections and suppress all classes except person and car
        bboxes = postprocess(detections, img_mat.shape[:2], config.INPUT_SIZE, threshold,
                             config.ANCHORS, config.STRIDES, config.XYSCALE)
        bboxes = suppress_classes_except(bboxes, [0, 2])

        # Convert detections to a JSON-format list
        bboxes_json = bboxes_to_json_list(bboxes, config.CLASS_NAMES)

        return {"num_detections": len(bboxes_json), "detections": bboxes_json}
    except HTTPException as e:
        # Re-raise the HTTPException to maintain consistent error responses
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

