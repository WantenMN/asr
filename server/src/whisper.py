from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import uvicorn
import zhconv

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "models/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

app = FastAPI()


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()

        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(audio_data)

        result = pipe(
            temp_file_path,
            generate_kwargs={"language": "zh"},
        )
        text = result.get("text", "")
        return JSONResponse(content={"text": zhconv.convert(text, "zh-cn")})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
