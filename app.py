import numpy as np
import cv2
import onnxruntime
import sys
from PIL import Image
import os

def pre_process(img: np.array) -> np.array:
    try:
        # H, W, C -> C, H, W
        img = np.transpose(img[:, :, 0:3], (2, 0, 1))
        # C, H, W -> 1, C, H, W
        img = np.expand_dims(img, axis=0).astype(np.float32)
    except Exception as e:
        print(f"Error in pre-processing the image: {e}")
        sys.exit(1)
    return img

def post_process(img: np.array) -> np.array:
    try:
        # 1, C, H, W -> C, H, W
        img = np.squeeze(img)
        # C, H, W -> H, W, C
        img = np.transpose(img, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
    except Exception as e:
        print(f"Error in post-processing the image: {e}")
        sys.exit(1)
    return img

def inference(model_path: str, img_array: np.array) -> np.array:
    try:
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        ort_session = onnxruntime.InferenceSession(model_path, options)
        ort_inputs = {ort_session.get_inputs()[0].name: img_array}
        ort_outs = ort_session.run(None, ort_inputs)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
    return ort_outs[0]

def convert_pil_to_cv2(image):
    try:
        open_cv_image = np.array(image)
        # RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
    except Exception as e:
        print(f"Error converting PIL image to OpenCV format: {e}")
        sys.exit(1)
    return open_cv_image


def upscale(image_path, model):
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_directory, "models", f"{model}.ort")
        image = Image.open(image_path)
    except FileNotFoundError:
        print("Error: Image file not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image or model: {e}")
        sys.exit(1)

    img = convert_pil_to_cv2(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3]  # GRAY
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # BGR
        alpha_output = post_process(inference(model_path, pre_process(alpha)))  # BGR
        alpha_output = cv2.cvtColor(alpha_output, cv2.COLOR_BGR2GRAY)  # GRAY
        img = img[:, :, 0:3]  # BGR
        image_output = post_process(inference(model_path, pre_process(img)))  # BGR
        image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2BGRA)  # BGRA
        image_output[:, :, 3] = alpha_output
    elif img.shape[2] == 3:
        image_output = post_process(inference(model_path, pre_process(img)))  # BGR
    return image_output

if __name__ == "__main__":
    try:
        image_path = "7ef47509-61fb-4db6-9781-de2e3d646185_resized.png"
        model_name = "/models/minecraft_modelx4.ort"
        output_image = upscale(image_path, model_name)
        cv2.imwrite("upscaled_image.png", output_image)
        print("Upscaling completed. Result saved as 'upscaled_image.png'.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
