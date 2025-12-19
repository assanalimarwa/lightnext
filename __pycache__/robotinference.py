from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "tinyllama/TinyLlama-1.1B-intermediate-step-1431k"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load small model for on-device inference
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

def generate_reply(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.2,
            do_sample=False
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


print(generate_reply("Hello robot, how do you feel today?"))



from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import bitsandbytes as bnb

MODEL_NAME = "tinyllama/TinyLlama-1.1B-intermediate-step-1431k"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    quantization_config=bnb.QuantizationConfig(
        load_in_4bit=True,
        dtype=torch.float16,
        quant_type="nf4"
    )
)

def infer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output[0])

print(infer("Explain what a robot does in simple words."))



import asyncio
from queue import Queue
from threading import Thread

class LLMRuntime:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.input_queue = Queue()
        self.output_queue = Queue()
        Thread(target=self.worker_loop, daemon=True).start()

    def worker_loop(self):
        while True:
            prompt = self.input_queue.get()
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=80
                )
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.output_queue.put(text)

    async def ask(self, text):
        self.input_queue.put(text)
        while self.output_queue.empty():
            await asyncio.sleep(0.01)
        return self.output_queue.get()

# Example usage:
# answer = await runtime.ask("Hello!")



import sounddevice as sd
import numpy as np
from transformers import pipeline

# Speech Recognition Model (Kazakh if available)
stt = pipeline("automatic-speech-recognition", model="kz-asr-small")
tts = pipeline("text-to-speech", model="kazakh-tts-small")

def record_audio(seconds=3):
    audio = sd.rec(int(16000 * seconds), samplerate=16000, channels=1)
    sd.wait()
    return audio.squeeze()

def robot_conversation():
    print("Say something...")
    audio = record_audio(4)
    text = stt({"array": np.array(audio), "sampling_rate": 16000})["text"]

    llm_reply = generate_reply(text)

    audio_out = tts(llm_reply)["audio"]
    sd.play(audio_out, 22050)
    sd.wait()



