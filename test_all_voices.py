#!/usr/bin/env python3
"""Test all Pocket TTS voices - generate a sample for each."""
import time
import os
from pocket_tts import TTSModel
import scipy.io.wavfile

OUTPUT_DIR = "/opt/data/workspace/projects/pocket-tts/voice_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All voices from the README
ALL_VOICES = {
    # English voices
    "alba": ("en", "Hey there! I'm Alba, nice to meet you."),
    "anna": ("en", "Hello, I'm Anna. Welcome to Pocket TTS."),
    "azelma": ("en", "Hi, my name is Azelma, let's chat."),
    "bill_boerst": ("en", "Greetings! Bill Boerst here, ready to talk."),
    "caro_davy": ("en", "Hi there! I'm Caro Davy, pleasure to speak with you."),
    "charles": ("en", "Good day, I'm Charles. Let me tell you something interesting."),
    "cosette": ("en", "Hello! Cosette speaking, what a wonderful day."),
    "eponine": ("en", "Hey, I'm Eponine. Let me share a thought with you."),
    "eve": ("en", "Hi, I'm Eve. The future of text to speech is here."),
    "fantine": ("en", "Hello, Fantine here. I love talking to people."),
    "george": ("en", "Good morning! George speaking. Let's get started."),
    "jane": ("en", "Hi there, I'm Jane. Always happy to help."),
    "jean": ("en", "Hello, Jean here. Ready for a conversation."),
    "javert": ("en", "Greetings, I'm Javert. The law is the law."),
    "marius": ("en", "Hey! Marius here. Let me tell you a story."),
    "mary": ("en", "Hello, I'm Mary. It's lovely to speak with you."),
    "michael": ("en", "Hi, Michael here. Technology is amazing these days."),
    "paul": ("en", "Good day! Paul speaking. Let's explore together."),
    "peter_yearsley": ("en", "Hello, Peter Yearsley here. Great to be heard."),
    "stuart_bell": ("en", "Hi there, Stuart Bell here. Let's do this."),
    "vera": ("en", "Hello! I'm Vera. Isn't voice technology wonderful?"),
    # Non-English voices
    "giovanni": ("it", "Ciao! Sono Giovanni, piacere di conoscerti."),
    "lola": ("es", "Hola! Soy Lola, encantada de hablar contigo."),
    "juergen": ("de", "Hallo! Ich bin Juergen, freut mich Sie kennenzulernen."),
    "rafael": ("pt", "Olá! Eu sou Rafael, prazer em falar com você."),
    "estelle": ("fr", "Bonjour! Je suis Estelle, ravie de vous parler."),
}

print(f"Loading Pocket TTS model...")
start = time.time()
model = TTSModel.load_model()
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s\n")

results = []
total_voices = len(ALL_VOICES)

for i, (voice_name, (lang, text)) in enumerate(ALL_VOICES.items(), 1):
    print(f"[{i}/{total_voices}] Generating: {voice_name} ({lang})...", end=" ", flush=True)
    try:
        t0 = time.time()
        voice_state = model.get_state_for_audio_prompt(voice_name)
        audio = model.generate_audio(voice_state, text)
        gen_time = time.time() - t0
        
        output_path = os.path.join(OUTPUT_DIR, f"{voice_name}.wav")
        scipy.io.wavfile.write(output_path, model.sample_rate, audio.numpy())
        
        duration = len(audio) / model.sample_rate
        rtf = gen_time / duration if duration > 0 else 0
        results.append((voice_name, lang, gen_time, duration, rtf, "OK"))
        print(f"OK ({gen_time:.1f}s gen, {duration:.1f}s audio, RTF={rtf:.2f}x)")
    except Exception as e:
        results.append((voice_name, lang, 0, 0, 0, str(e)))
        print(f"FAILED: {e}")

# Summary
print("\n" + "=" * 70)
print("POCKET TTS - ALL VOICES TEST RESULTS")
print("=" * 70)
print(f"{'Voice':<20} {'Lang':<5} {'Gen(s)':<8} {'Audio(s)':<9} {'RTF':<7} {'Status'}")
print("-" * 70)
ok_count = 0
for voice, lang, gen_t, dur, rtf, status in results:
    st = "✅" if status == "OK" else "❌"
    if status == "OK":
        ok_count += 1
        print(f"{voice:<20} {lang:<5} {gen_t:<8.1f} {dur:<9.1f} {rtf:<7.2f} {st}")
    else:
        print(f"{voice:<20} {lang:<5} {'--':<8} {'--':<9} {'--':<7} {st} {status[:40]}")

print("-" * 70)
print(f"Total: {ok_count}/{total_voices} voices succeeded")
print(f"Output directory: {OUTPUT_DIR}")
