from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import uuid
import cv2
import rawpy
import numpy as np
from skimage.filters import unsharp_mask
from skimage.color import rgb2hsv, hsv2rgb
from scipy import ndimage
import imutils
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from huggingface_hub import InferenceClient
import io
import uuid
#import random
#from diffusers import DiffusionPipeline
#from diffusers import FluxKontextPipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# The rest of your code as is
# Create the Hugging Face API client ONCE
# Create the Hugging Face API client ONCE
HF_TOKEN = os.environ.get("HF_TOKEN") # 
if not HF_TOKEN:
    raise RuntimeError("Please set the HF_TOKEN environment variable.")

client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)

@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        # Read the input image bytes
        image_bytes = await file.read()

        # Send to Hugging Face Inference API with the correct model for image-to-image
        # You need to replace 'fal-ai/flux/dev/image-to-image' with the exact model ID
        # that supports image-to-image with Fal.ai and matches your intended use case.
        # For 'black-forest-labs/FLUX.1-Kontext-dev', you'll need to find its specific
        # image-to-image endpoint if one exists and is different from text-to-image.
        result = client.image_to_image(
            image_bytes,
            prompt=prompt,
            model="black-forest-labs/FLUX.1-Kontext-dev", # <--- **FIX THIS LINE**
            return_type="pil"  # This makes result always PIL.Image
        )

        # Save the result or return it
        # For example, to save and then return as a response:
        output_buffer = io.BytesIO()
        result.save(output_buffer, format="PNG") # or JPEG, depending on your needs
        output_buffer.seek(0)

        # You'll likely want to return the enhanced image, not just a success message
        return JSONResponse(content={"message": "Image enhanced successfully!"}, status_code=200)

    except RuntimeError as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)


# Serve index.html if you wish
from fastapi.responses import HTMLResponse
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def professional_enhance(image):
    # 1. White Balance
    image = cv2.xphoto.createSimpleWB().balanceWhite(image)

    # 2. CLAHE (contrast boost)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 3. Gamma correction (brighten gently)
    gamma = 1.25
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0,256)]).astype("uint8")
    image = cv2.LUT(image, table)

    # 4. Denoise
    image = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)

    # 5. Sharpen
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image = cv2.addWeighted(image, 1.20, blurred, -0.18, 0)

    # 6. Slight vibrance boost
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[...,1] = cv2.add(hsv[...,1], 10)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 7. Tiny extra brightness
    image = cv2.convertScaleAbs(image, alpha=1.04, beta=6)

    return image

# === Optional: Keep your helpers for future toggles or features ===

def auto_white_balance_correction(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def enhance_contrast_brightness(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_sharpness_clarity(image):
    return unsharp_mask(image, radius=2, amount=1.5, preserve_range=True).astype(np.uint8)

def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(
        np.array(screenCnt.reshape(4, 2)), 
        np.array([(0, 0), (image.shape[1], 0), 
                 (image.shape[1], image.shape[0]), (0, image.shape[0])])), 
        (image.shape[1], image.shape[0]))

def adjust_vibrance(image, amount=1.5):
    hsv = rgb2hsv(image)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * amount, 0, 1)
    return hsv2rgb(hsv) * 255

def apply_tone_mapping(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def enhance_details(image):
    kernel = np.array([[-1, -1, -1], 
                       [-1, 9, -1], 
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def correct_chromatic_aberration(image):
    b, g, r = cv2.split(image)
    aligned_g = ndimage.shift(g, (0, 0), order=0)
    aligned_r = ndimage.shift(r, (0, 0), order=0)
    return cv2.merge((b, aligned_g, aligned_r))

def apply_vignette(image, amount=0.8):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask[:, :, np.newaxis]
    return np.uint8(np.clip(image * (1 - amount * (1 - mask/255.0)), 0, 255))

def adjust_shadows_highlights(image, shadows=0.5, highlights=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    ret, shadows_mask = cv2.threshold(l, 50, 255, cv2.THRESH_BINARY_INV)
    shadows_mask = cv2.GaussianBlur(shadows_mask, (0,0), 3)
    l = np.uint8(np.clip(l + shadows * shadows_mask/255.0 * 50, 0, 255))
    ret, highlights_mask = cv2.threshold(l, 200, 255, cv2.THRESH_BINARY)
    highlights_mask = cv2.GaussianBlur(highlights_mask, (0,0), 3)
    l = np.uint8(np.clip(l - highlights * highlights_mask/255.0 * 50, 0, 255))
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def align_images(images):
    import cv2
    import numpy as np
    print("[DEBUG] Aligning images with ECC...")
    aligned = [images[0]]
    ref_img = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    for idx, img in enumerate(images[1:], 1):
        print(f"[DEBUG] Aligning image {idx+1} of 3")
        im1_gray = ref_img
        im2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sz = img.shape
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        aligned_img = cv2.warpAffine(img, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        aligned.append(aligned_img)
    print("[DEBUG] Alignment complete.")
    return aligned


def merge_hdr_debevec(images, exposure_times):
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times=np.array(exposure_times, dtype=np.float32))
    # Tonemap to 8-bit for display/export
    tonemap = cv2.createTonemap(2.2)
    ldr = tonemap.process(hdr)
    ldr_8bit = np.clip(ldr*255, 0, 255).astype('uint8')
    return ldr_8bit

def merge_hdr_mertens(images):
    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(images)
    ldr_8bit = np.clip(fusion*255, 0, 255).astype('uint8')
    return ldr_8bit


# === Process Endpoint: Always Use Pro Enhancement Pipeline ===
@app.post("/process")
async def process_images(
    files: List[UploadFile] = File(...),
):
    print("[DEBUG] /process endpoint called.")
    try:
        print(f"[DEBUG] Received {len(files)} file(s) for processing.")

        processed_images = []
        file_paths = []

        # Save all uploaded files
        for file in files:
            file_ext = os.path.splitext(file.filename)[1]
            file_name = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join(UPLOAD_DIR, file_name)
            with open(file_path, "wb") as buffer:
                file_bytes = await file.read()
                buffer.write(file_bytes)
            print(f"[DEBUG] Saved upload: {file_path} ({len(file_bytes)} bytes)")
            file_paths.append(file_path)

        # HDR merge if exactly 3 images
        if len(file_paths) == 3:
            print("[DEBUG] Attempting HDR merge pipeline.")
            images = []
            for fp in file_paths:
                print(f"[DEBUG] Reading image: {fp}")
                if fp.lower().endswith(('.nef', '.cr2', '.arw', '.dng')):
                    with rawpy.imread(fp) as raw:
                        rgb = raw.postprocess()
                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.imread(fp)
                if img is None:
                    print(f"[ERROR] Could not read {fp}.")
                    raise HTTPException(status_code=400, detail=f"Invalid image: {fp}")
                print(f"[DEBUG] Image shape: {img.shape}")
                images.append(img)
            print("[DEBUG] All images read, starting alignment.")
            exposure_times = [1/30.0, 0.25, 2.5]
            images_aligned = align_images(images)
            print("[DEBUG] Alignment done, starting HDR merge.")
            hdr_ldr = merge_hdr_debevec(images_aligned, exposure_times)
            mertens_ldr = merge_hdr_mertens(images_aligned)
            print("[DEBUG] HDR and Mertens merges done, starting enhancement.")
            hdr_final = professional_enhance(hdr_ldr)
            mertens_final = professional_enhance(mertens_ldr)
            processed_hdr_jpg = f"processed_HDR_{uuid.uuid4().hex}.jpg"
            processed_mertens_jpg = f"processed_Mertens_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(os.path.join(UPLOAD_DIR, processed_hdr_jpg), hdr_final)
            cv2.imwrite(os.path.join(UPLOAD_DIR, processed_mertens_jpg), mertens_final)
            processed_hdr_tiff = f"processed_HDR_{uuid.uuid4().hex}.tiff"
            cv2.imwrite(os.path.join(UPLOAD_DIR, processed_hdr_tiff), hdr_final)
            print("[DEBUG] HDR processing complete. Returning files to frontend.")

            return JSONResponse({
                "success": True,
                "processed_files": [
                    f"/uploads/{processed_hdr_jpg}",
                    f"/uploads/{processed_mertens_jpg}",
                    f"/uploads/{processed_hdr_tiff}",
                ],
                "mode": "hdr"
            })

        # Single image enhancement
        else:
            for file_path in file_paths:
                print(f"[DEBUG] Processing single image: {file_path}")
                if file_path.lower().endswith(('.nef', '.cr2', '.arw', '.dng')):
                    with rawpy.imread(file_path) as raw:
                        rgb = raw.postprocess()
                    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    image = cv2.imread(file_path)
                if image is None:
                    print(f"[ERROR] Could not read {file_path}.")
                    raise HTTPException(status_code=400, detail=f"Invalid image: {file_path}")
                print(f"[DEBUG] Image shape: {image.shape}")
                enhanced = professional_enhance(image)
                processed_name = f"processed_{os.path.basename(file_path)}"
                processed_path = os.path.join(UPLOAD_DIR, processed_name)
                cv2.imwrite(processed_path, enhanced)
                print(f"[DEBUG] Saved processed image: {processed_path}")
                processed_images.append(f"/uploads/{processed_name}")

            print("[DEBUG] Single image enhancement done. Returning files to frontend.")
            return JSONResponse({
                "success": True,
                "processed_files": processed_images,
                "mode": "single"
            })

    except Exception as e:
        print(f"[EXCEPTION] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reprocess")
async def reprocess_image(
    image_url: str = Form(...),
    brightness: int = Form(...),
    contrast: int = Form(...),
    saturation: int = Form(...),
):
    import cv2
    import numpy as np
    # Get the image path from image_url (strip /uploads/ etc. as needed)
    image_path = image_url.replace("/uploads/", "uploads/")
    image = cv2.imread(image_path)
    # Apply adjustments
    # Brightness and contrast
    image = cv2.convertScaleAbs(image, alpha=1 + int(contrast)/100, beta=int(brightness))
    # Saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[...,1] = cv2.add(hsv[...,1], int(saturation))
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Save
    out_path = "uploads/reprocessed_" + os.path.basename(image_path)
    cv2.imwrite(out_path, image)
    return {"processed_file": f"/uploads/{os.path.basename(out_path)}"}


# === Sample Demo Images (optional, for preview) ===
@app.get("/sample_before.jpg")
async def get_sample_before():
    return FileResponse("sample_images/sample_before.jpg")

@app.get("/sample_after.jpg")
async def get_sample_after():
    return FileResponse("sample_images/sample_after.jpg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    #uvicorn main:app --reload
