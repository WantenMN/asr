from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from funasr_onnx import Paraformer
import uvicorn
import zhconv
import os

model_id = "/home/wanten/repos/speech-recognition/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"

model = Paraformer(model_dir=model_id, batch_size=1, quantize=True)

app = FastAPI()


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()

        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(audio_data)

        result = model(temp_file_path)
        print(result)
        text = result[0]["preds"][0]
        os.remove(temp_file_path)

        return JSONResponse(content={"text": zhconv.convert(text, "zh-cn")})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
