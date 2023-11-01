from fastapi import FastAPI, File, UploadFile
import onnxruntime as rt
import cv2
import numpy as np
from src.yolo_utils import image_preprocess, postprocess, suppress_classes_except, bboxes_to_json_list
import src.config as config
import json

# load the model
model = rt.InferenceSession("yolov4.onnx")

# start the app
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


# receives image and returns bounding boxes as json
@app.post("/detect")
async def detect(image: UploadFile = File(...)):

    # access image as numpy array
    request_object_content = await image.read()
    image_array = np.fromstring(request_object_content, np.uint8)
    img_mat = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    # preprocess image
    preprocessed_img = image_preprocess(img_mat, (config.INPUT_SIZE, config.INPUT_SIZE))
    # add batch dimension
    preprocessed_img = preprocessed_img[np.newaxis, ...].astype(np.float32)

    # run inference
    outputs = model.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = model.get_inputs()[0].name
    detections = model.run(output_names, {input_name: preprocessed_img})

    # postprocess detections and suppress all classes except person and car
    bboxes = postprocess(detections, img_mat.shape[:2], config.INPUT_SIZE, config.THRESHOLD,
                         config.ANCHORS, config.STRIDES, config.XYSCALE)
    bboxes = suppress_classes_except(bboxes, [0, 2])

    # convert detections to json-format list
    bboxes_json = bboxes_to_json_list(bboxes, config.CLASS_NAMES)

    return {"num_detections": len(bboxes_json), "detections": bboxes_json}

