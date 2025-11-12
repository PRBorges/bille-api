from ultralytics import YOLO
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io

# Modelos para el server
models = {
    "USD": YOLO("models/usd-best.pt"),
    "VEF": YOLO("models/ves-best.pt"),
    "INFERENCIA": YOLO("models/currency-best.pt"),
}

# Clases de los modelos
classes = {
    "USD": [
        "fifty-back",
        "fifty-front",
        "five-back",
        "five-front",
        "one-back",
        "one-front",
        "one_hundred-back",
        "one_hundred-front",
        "ten-back",
        "ten-front",
        "twenty-back",
        "twenty-front",
    ],
    "VEF": [
        "fifty-back-vef",
        "fifty-front-vef",
        "five-back-vef",
        "five-front-vef",
        "one_hundred-back-vef",
        "one_hundred-front-vef",
        "ten-back-vef",
        "ten-front-vef",
        "twenty-back-vef",
        "twenty-front-vef",
        "two_hundred-back-vef",
        "two_hundred-front-vef",
    ],
    "INFERENCIA": [
        "dollar_back",
        "dollar_front",
        "vef_back",
        "vef_front",
    ],
}


# Creamos la app del FastAPI
app = FastAPI()
# Se agrega el CORS para aceptar peticiones de un origen externo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detection")
async def detection(image: UploadFile):
    image_bytes = await image.read()
    image_stream = io.BytesIO(image_bytes)
    image_file = Image.open(image_stream)

    which_currency = models["INFERENCIA"].predict(image_file, verbose=False)
    if len(which_currency[0].boxes) == 0:
        return {"message": "No objects detected"}
    currency_label = classes["INFERENCIA"][int(which_currency[0].boxes[0].cls.item())]
    if "vef" in currency_label:
        currency = "VEF"
    else:
        currency = "USD"
    results = models[currency].predict(
        image_file, verbose=False
    )  # Se pasa la imagen por el modelo

    # Se extraen las cajas de los resultados
    if len(results[0].boxes) > 0:
        boxes = []
        for box in results[0].boxes:
            boxes.append(
                {
                    "label": classes[currency][int(box.cls.item())],  # Clase detectada
                    "confidence": box.conf.item(),  # Confianza
                    "bbox": box.xyxy.tolist(),  # Coordenadas del cuadro
                }
            )
        return {"detections": boxes}
    else:
        return {"message": "No objects detected"}
