import os
import subprocess
import tempfile
import torch
import whisper
import csv
import json
import tempfile
from datetime import timedelta
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
import string
from pyannote.audio import Pipeline
from pyannote.core import Segment
from collections import defaultdict
import torchaudio

import warnings
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels.*")
# from pyannote.audio.utils.reproducibility import ReproducibilityWarning
# warnings.filterwarnings("ignore", category=ReproducibilityWarning)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load Whisper model (local)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
### whisper model
model = whisper.load_model("large-v3").to(device)  # or "tiny" for speed, ## or "base" for even faster

diarizer = Pipeline.from_pretrained(
                checkpoint_path="pyannote/speaker-diarization",  #pyannote/speaker-diarization-3.1
                use_auth_token=os.environ["HF_TOKEN"],  # Set HF_TOKEN in the environment.
            ).to(torch.device("cuda"))

lemmatizer = WordNetLemmatizer()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

number_words = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10"
}


def extract_main_speaker_segments(audio_path, all_speakers=True):
    waveform, sr = torchaudio.load(audio_path)
    diarization = diarizer({"waveform": waveform, "sample_rate": sr})

    durations = defaultdict(float)
    segments_by_speaker = defaultdict(list)

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        duration = segment.end - segment.start
        durations[speaker] += duration
        segments_by_speaker[speaker].append(segment)

    if not durations:
        return []

    main_speaker = max(durations, key=durations.get)
    print(f"Main speaker is extracted")
    if all_speakers:
        return durations, segments_by_speaker  # Return all speakers' segments
        # print(f"Main speaker: {main_speaker} with duration: {durations[main_speaker]:.2f} seconds")
    else:
        return durations, segments_by_speaker[main_speaker]


# Prepare category embeddings
def build_category_embeddings(categories):
    texts = [f"{c['name']} {c.get('description', '')} {c.get('abbreviation', '')}" for c in categories]
    return embedder.encode(texts, convert_to_tensor=True)

# Load and process category CSV
csv_path = "D:\\trash_dataset\\material_classifications\\Sorting Categories.csv"
df = pd.read_csv(csv_path)

def get_categories(df):
    # Strip and rename columns
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        "Major Category": "supercategory",
        "Sorting Category": "name",
        "Abbreviation": "abbreviation",
        "Consolidated Category": "recycle_category",
        "Sorting Category Brief Definition": "description"
    })

    # Forward fill 'supercategory' for rows with missing values
    # df["supercategory"] = df["supercategory"].fillna(method="ffill")
    df["supercategory"] = df["supercategory"].ffill()
    # Remove unnecessary columns
    df = df[["supercategory", "name", "abbreviation", "recycle_category", "description"]]
    df["abbreviation"] = df["abbreviation"].fillna("").astype(str)

    df = df.where(pd.notnull(df), None)

    # Convert to list of dicts for JSON generation
    categories = df.to_dict(orient="records")
    # categories[:3]  # preview
    return categories


categories = get_categories(df)
category_embeddings = build_category_embeddings(categories)

def normalize(text):
    """Lowercase, remove punctuation, and lemmatize"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    # text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase to space -> ex: Phrase with spaces: "This is an example" Lower camel case: "thisIsAnExample" Upper camel case (Pascal case): "ThisIsAnExample" 
    if "boxboard" in text:
        text = text.replace("boxboard", "box board")
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(w, pos='n') for w in words]
    return ' '.join(lemmatized)

def normalize_phrase(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    # Replace number words
    # if text.strip() == "3 to 7":
    #     return text  # skip phrases with hyphens (e.g., "one-two")
    # Skip numeric ranges like "3 to 7"
    # if re.search(r"\b\d+\s+to\s+\d+\b", text):
    #     return text
    tokens = text.split()
    tokens = [number_words.get(t, t) for t in tokens]
    # print(f"Normalized phrase: {text} -> {' '.join(tokens)}")
    return " ".join(tokens)


def match_category(text, method="substring", threshold=90):
    norm_text = normalize(text)

    best_match = None
    best_score = 0

    for cat in categories:
        name = normalize(cat['name'] or "")
        desc = normalize(cat['description'] or "")

        if method == "substring":
            if name in norm_text or desc in norm_text:
                return cat["name"]

        elif method == "fuzzy":
            score_name = fuzz.partial_ratio(norm_text, name)
            score_desc = fuzz.partial_ratio(norm_text, desc)
            score = max(score_name, score_desc)

            if score > best_score and score > threshold:
                best_score = score
                best_match = cat["name"]

    return best_match or None

def match_category_semantic(phrase, categories, threshold=0.5):
    phrase_embedding = embedder.encode(phrase, convert_to_tensor=True)
    similarities = util.cos_sim(phrase_embedding, category_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    return categories[best_idx]["name"] if best_score >= threshold else None


def smart_match_category(phrase, categories, threshold=0.2):
    phrase_lower = phrase.lower()

    # 1. Exact Substring Match (Sorting Category / name)
    for cat in categories:
        name = cat.get("name", "")
        name = normalize_phrase(name)  # Normalize name
        if name in phrase_lower or name.lower() in phrase_lower:
            # print(f"Exact match found in sorting categories: {name} for phrase: {phrase}")
            return name

    # 2. Abbreviation Match
    for cat in categories:
        abbr = cat.get("abbreviation", "")
        # print(f"Abbreviation: {abbr}")
        abbr = normalize_phrase(abbr)  # Normalize abbreviation
        if not abbr:  # Skip empty abbreviations
            continue
        if abbr in phrase_lower or abbr.lower() in phrase_lower:
            # print(f"Abbreviation match found: {abbr} for phrase: {phrase} -> {cat.get('name')}")
            return cat.get("name")

    # 3. Fallback: Semantic Matching
    phrase_embedding = embedder.encode(phrase, convert_to_tensor=True)
    similarities = util.cos_sim(phrase_embedding, category_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    # if "3 to 7" in phrase:
    # print(f"Semantic match {categories[best_idx]['name']} for phrase: {phrase} — Score: {best_score:.3f}")

    return categories[best_idx]["name"] if best_score >= threshold else None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_prompt_from_categories(categories: list[str]) -> str:
    unique_names = {cat['name'] for cat in categories if cat['name']}
    return ", ".join(sorted(unique_names))


initial_prompt = get_prompt_from_categories(categories)


def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    return float(subprocess.check_output(cmd).decode().strip())

def extract_dual_audio_ffmpeg(video_path, start_time, end_time, base_name, wav_dir, mp3_dir):
    ensure_dir(wav_dir)
    ensure_dir(mp3_dir)

    wav_path = os.path.join(wav_dir, f"{base_name}.wav")
    mp3_path = os.path.join(mp3_dir, f"{base_name}.mp3")

    for fmt, path, out_dir in [("wav", wav_path, wav_dir), ("mp3", mp3_path, mp3_dir)]:
        cmd = [
            "ffmpeg", "-ss", str(start_time), "-to", str(end_time),
            "-i", video_path, "-vn",
            "-ar", "16000", "-ac", "1", "-f", fmt, path, "-y"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return wav_path, mp3_path

def extract_audio_ffmpeg(video_path, start_time, end_time):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        audio_path = temp_audio.name

    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        "-vn",
        "-ar", "16000",    # 16 kHz
        "-ac", "1",        # Mono
        "-f", "wav",
        audio_path,
        "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path, language="en", initial_prompt=initial_prompt)
        text = result['text'].strip()
        return text if text else None
    except Exception as e:
        return None
    # finally:
    #     os.remove(audio_path)

def extract_audio_segments_ffmpeg(video_path, output_csv, wav_dir, mp3_dir, interval_sec, overlap_sec):
    duration = get_video_duration(video_path)

    records = []
    save_idx = 0
    step_size = interval_sec - overlap_sec  # e.g., 9 - 1 = 8
    start = 0.0

    while start < duration:
        end = min(start + interval_sec, duration)
        base_name = f"frame_{save_idx}"
        wav_path, _ = extract_dual_audio_ffmpeg(video_path, start, end, base_name, wav_dir, mp3_dir)
        transcript = transcribe_audio(wav_path)

        # audio_file = extract_audio_ffmpeg(video_path, start, end)
        # transcript = transcribe_audio(audio_file)

        if transcript:  # Skip empty/silent clips
            records.append([
                base_name,
                str(timedelta(seconds=round(start))),
                str(timedelta(seconds=round(end))),
                transcript,
                match_category(transcript, method="substring"),  # or method="substring"
                # extract_label_from_text(transcript)
            ])
            save_idx += 1
        start += step_size # interval_sec

    # Save CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Chunk_ID', 'Start_Time', 'End_Time', 'Transcript', 'Object_Label'])
        writer.writerows(records)


def save_full_audio(video_path, output_dir_wav, output_dir_mp3):
    ensure_dir(output_dir_wav)
    ensure_dir(output_dir_mp3)
    wav_path = os.path.join(output_dir_wav, "full_audio.wav")
    mp3_path = os.path.join(output_dir_mp3, "full_audio.mp3")

    for fmt, path in [("wav", wav_path), ("mp3", mp3_path)]:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-ar", "16000", "-ac", "1",
            "-f", fmt, path, "-y"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_segments_with_smart_phrases(segments, diarization_timeline):
    split_phrases = []
    for seg in segments:
        words = seg.get("words", [])
        phrase = ""
        start_time = None

        for i, w in enumerate(words):
            word = w["word"].strip()
            if start_time is None:
                start_time = w["start"]

            if word.lower() in ["uh", "um", "hmm", "ah"]:
                if phrase.strip():  # finish current phrase
                    split_phrases.append(build_phrase_dict(phrase.strip(), start_time, w["start"], diarization_timeline))
                split_phrases.append(build_phrase_dict("uh", w["start"], w["end"], diarization_timeline))
                phrase = ""
                start_time = None
                continue

            phrase += word + " "

            # End phrase if "." or "?" or last word
            if word.endswith((".", "?", "!", ",")) or i == len(words) - 1:
                split_phrases.append(build_phrase_dict(phrase.strip(), start_time, w["end"], diarization_timeline))
                phrase = ""
                start_time = None
    # print(f"split_phrases: {split_phrases}")
    return split_phrases


def build_phrase_dict(phrase, start_time, end_time, diarization_timeline):
    lower = phrase.lower()

    # Skip question or negated sentences
    if "?" in phrase or re.match(r"^(who|what|where|why|is|are|can|does|do|should|would)\b", lower):
        cat_name = None
    # # Heuristics: skip if mostly filler or repetition
    # elif len(set(lower_phrase.split())) <= 3:
    #     cat_name = None
    elif re.search(r"\bnot\b|\bisnt\b|\bdoesnt\b|\bno\b", lower):
        cat_name = None
    else:
        cat_name = smart_match_category(phrase, categories, threshold=0.2)
        # print(f"in :::::::build_phrase_dict::::::: Matched category: {cat_name} for phrase: {phrase}")

    # # 🔄 Assign speaker based on diarization timeline
    # matched_speakers = {
    #     entry["speaker"] for entry in diarization_timeline
    #     if not (entry["end"] < start_time or entry["start"] > end_time)
    # }
    # matched_speaker = sorted(matched_speakers)[0] if matched_speakers else None

    speaker_overlap = defaultdict(float)
    for entry in diarization_timeline:
        overlap_start = max(start_time, entry["start"])
        overlap_end = min(end_time, entry["end"])
        if overlap_start < overlap_end:
            speaker_overlap[entry["speaker"]] += (overlap_end - overlap_start)

    matched_speaker = max(speaker_overlap, key=speaker_overlap.get) if speaker_overlap else None
    
    return {
        "begin_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "phrase": phrase,
        "speaker_id": matched_speaker,  
        "image_id": None,
        "category_name": cat_name,
        "bbox": None,
        "segmentation": None
    }


def refine_json_categories_iteratively(json_path, thresholds, reevaluate=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified = False

    for threshold in thresholds:
        print(f"\n🔁 Running refinement pass with threshold={threshold:.2f}")
        for entry in data["time_stamped_text"]:
            phrase = entry["phrase"].strip()
            current_cat = entry.get("category_name", None)

            # CASE 1: Fill empty category
            if current_cat is None:
                # Step 1️⃣ — Try abbreviation match
                abbr_matched = False
                for cat in categories:
                    abbr = (cat.get("abbreviation") or "").lower()
                    if abbr and abbr in phrase.lower():
                        entry["category_name"] = cat["name"]
                        modified = True
                        abbr_matched = True
                        break

                if abbr_matched:
                    continue  # don't do semantic if already matched

                # Step 2️⃣ — Semantic match
                new_cat = match_category_semantic(phrase, categories, threshold=threshold)
                if new_cat:
                    entry["category_name"] = new_cat
                    modified = True

            # CASE 2: Re-evaluate existing category
            elif reevaluate:
                new_cat = match_category_semantic(phrase, categories, threshold=threshold)
                if new_cat != current_cat:
                    print(f"  🔄 Replacing '{current_cat}' → '{new_cat}' for phrase: '{phrase}'")
                    entry["category_name"] = new_cat
                    modified = True

    if modified:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Updated categories written to {json_path}")
    else:
        print("✅ No updates needed — all categories are set.")


def split_into_atomic_segments(time_stamped_text):
    """
    Split overlapping time-stamped segments into atomic segments, preserving multiple speakers.
    """
    time_points = set()
    for entry in time_stamped_text:
        time_points.add(entry["begin_time"])
        time_points.add(entry["end_time"])

    sorted_times = sorted(time_points)

    atomic_segments = []
    for i in range(len(sorted_times) - 1):
        start = sorted_times[i]
        end = sorted_times[i + 1]

        active_entries = [
            entry for entry in time_stamped_text
            if not (entry["end_time"] <= start or entry["begin_time"] >= end)
        ]

        if not active_entries:
            continue

        speakers = list({entry["speaker_id"] for entry in active_entries})
        phrases = " ".join(entry["phrase"] for entry in active_entries)
        categories = list({entry["category_name"] for entry in active_entries if entry["category_name"]})

        atomic_segments.append({
            "begin_time": round(start, 2),
            "end_time": round(end, 2),
            "phrase": phrases.strip(),
            "speaker_ids": speakers,
            "category_names": categories if categories else None,
            "image_id": None,
            "bbox": None,
            "segmentation": None
        })

    return atomic_segments


def match_speaker_from_diarization(start, end, diarization_timeline):
    from collections import defaultdict
    overlaps = defaultdict(float)
    for entry in diarization_timeline:
        overlap = max(0, min(end, entry["end"]) - max(start, entry["start"]))
        if overlap > 0:
            overlaps[entry["speaker"]] += overlap
    return max(overlaps, key=overlaps.get) if overlaps else "Unknown"


def transcribe_video_to_json_directly(video_path, output_json_path, all_speakers=True):
    print(f"Transcribing full video: {os.path.basename(video_path)}")
    
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1",
        "-f", "wav", audio_path, "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    waveform, sr = torchaudio.load(audio_path)

    # Step 2: Transcribe with Whisper
    result = model.transcribe(audio_path, language="en", word_timestamps=True)
    full_text = result["text"]
    segments_data = result.get("segments", [])

    diarization = diarizer({"waveform": waveform, "sample_rate": sr})

    # durations, speaker_segments = extract_main_speaker_segments(audio_path, all_speakers=all_speakers)
    # if not speaker_segments:
    #     print(f"No speaker segments found in audio {audio_path}. Skipping.")
    #     return

    # ✅ Speaker durations + mapping
    durations = defaultdict(float)
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        durations[speaker] += segment.end - segment.start

    if not durations:
        print(f"❌ No speaker segments found in {os.path.basename(video_path)}. Skipping.")
        return

    # NEW: Determine main speaker
    main_speaker_pyannote_id = max(durations, key=durations.get)
    speaker_mapping = {spk: f"Speaker {i+1}" for i, spk in enumerate(durations.keys())}
    main_speaker_id = speaker_mapping[main_speaker_pyannote_id]

    # NEW: assign speakers post-hoc to Whisper segments
    diarization_timeline = []
    for seg, _, spk in diarization.itertracks(yield_label=True):
        diarization_timeline.append({
            "start": seg.start,
            "end": seg.end,
            "speaker": speaker_mapping[spk]
        })
    
    time_stamped_text = parse_segments_with_smart_phrases(segments_data, diarization_timeline)

    main_speaker_text = " ".join(
        [seg["phrase"] for seg in time_stamped_text if seg["speaker_id"] == main_speaker_id]
    )  # 🔄 MODIFIED
    sorted_text = sorted(time_stamped_text, key=lambda x: x["begin_time"])

    data = {
        "full_text": full_text,
        "main_speaker_text": main_speaker_text,
        "main_speaker_id": main_speaker_id,
        "time_stamped_text": sorted_text,
        "categories": categories,
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved full transcription to {output_json_path}")    


def transcribe_full_video_to_json(video_path, output_json_path, all_speakers=True):
    print(f"Transcribing full video: {os.path.basename(video_path)}")
    
    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1",
        "-f", "wav", audio_path, "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    speaker_segments = extract_main_speaker_segments(audio_path, all_speakers=all_speakers)

    if not speaker_segments:
        print(f"No speaker segments found in audio {audio_path}. Skipping.")
        return    

    waveform, sr = torchaudio.load(audio_path)

    # Always compute main speaker transcription
    # main_speaker_id = max(speaker_segments, key=lambda s: sum(seg.end - seg.start for seg in speaker_segments[s]))
    # main_segments = speaker_segments[main_speaker_id]
    speaker_durations = {s: sum(seg.end - seg.start for seg in segs) for s, segs in speaker_segments.items()}
    main_speaker_pyannote_id = max(speaker_durations, key=speaker_durations.get)
    main_segments = speaker_segments[main_speaker_pyannote_id]

    # Step 2: Create speaker mapping to readable labels
    speaker_mapping = {}
    for idx, spk in enumerate(speaker_segments.keys(), start=1):
        speaker_mapping[spk] = f"Speaker {idx}"

    main_speaker_id = speaker_mapping[main_speaker_pyannote_id]

    main_waveform = torch.cat([
        waveform[:, int(seg.start * sr):int(seg.end * sr)]
        for seg in main_segments
    ], dim=1)
    result_main = model.transcribe(main_waveform.squeeze(), language="en", initial_prompt=initial_prompt)
    main_speaker_text = result_main["text"]

    time_stamped_text = []

    if all_speakers:
        for pyannote_id, segments in speaker_segments.items():
            speaker_waveform = torch.cat([
                waveform[:, int(seg.start * sr):int(seg.end * sr)]
                for seg in segments
            ], dim=1)
    
            result = model.transcribe(speaker_waveform.squeeze(), language="en", word_timestamps=True, initial_prompt=initial_prompt)
            # result = model.transcribe(video_path, language="en", word_timestamps=True, initial_prompt=initial_prompt) ## --> all the speakers are considered
            segments_data = result.get("segments", [])
            speaker_label = speaker_mapping[pyannote_id]
            first_segment_start = segments[0].start if segments else 0.0
            time_stamped_text.extend(parse_segments_with_smart_phrases(segments_data, speaker_id=speaker_label, offset=first_segment_start))

            print(f"\n🔍 Debug for {speaker_label}")
            print(f"Segment time range: {segments[0].start:.2f}s → {segments[-1].end:.2f}s (duration: {segments[-1].end - segments[0].start:.2f}s)")
            print(f"Transcribed segment word time range: {segments_data[0]['words'][0]['start']:.2f}s → {segments_data[-1]['words'][-1]['end']:.2f}s")
    
        full_text = " ".join([seg["phrase"] for seg in time_stamped_text]) # result['text']
    
    else:
        result = model.transcribe(main_waveform.squeeze(), language="en", word_timestamps=True, initial_prompt=initial_prompt)
        segments_data = result.get("segments", [])
        time_stamped_text = parse_segments_with_smart_phrases(segments_data, speaker_id=main_speaker_id)
        full_text = main_speaker_text # result["text"]

    # atomic_segments = split_into_atomic_segments(time_stamped_text) ##  gives very bad results
    sorted_text = sorted(time_stamped_text, key=lambda x: x["begin_time"])

    data = {
        "full_text": full_text,
        "main_speaker_text": main_speaker_text,
        "main_speaker_id": main_speaker_id,
        "time_stamped_text": sorted_text, #time_stamped_text,
        "categories": categories,
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Saved full transcription to {output_json_path}")

    # refine_json_categories_iteratively(
    #     json_path=output_json_path,
    #     thresholds=[0.89],
    #     reevaluate=False # True  # Set to True to re-evaluate existing categories
    # )


# def transcribe_full_video_to_json(video_path, output_json_path):
#     print(f"Transcribing full video: {os.path.basename(video_path)}")
#     result = model.transcribe(video_path, language="en", word_timestamps=True, initial_prompt=initial_prompt)
#     full_text = result['text']
#     segments = result.get("segments", [])

#     time_stamped_text = []

#     for seg in segments:
#         words = seg.get("words", [])
#         phrase = ""
#         start_time = None
#         #phrase = seg["text"].strip()
#         #cat_name = match_category(phrase, method="substring")  # or method="substring" "fuzzy"
#         # label = extract_label_from_text(phrase) ## --> # This function is not used in the current implementation; it is for the direct phase mapping
#         # cat_id = category_map.get(label) ## # This function is not used in the current implementation; it is for the direct phase mapping

#         for i, w in enumerate(words):
#             if start_time is None:
#                 start_time = w["start"]
#             phrase += w["word"] #+ " "

#             if w["word"].strip().endswith((".", "?", "!")) or i == len(words) - 1:
#                 end_time = w["end"]
#                 sub_phrase = phrase.strip()

#                 lower_phrase = sub_phrase.lower()

#                 # Heuristics: skip questions
#                 if "?" in sub_phrase or re.match(r"^(who|what|where|when|why|is|are|do|does|can|should)\b", lower_phrase):
#                     cat_name = None
                
#                 # # Heuristics: skip if mostly filler or repetition
#                 # elif len(set(lower_phrase.split())) <= 3:
#                 #     cat_name = None

#                 # Negation-aware filtering
#                 elif re.search(r'\bnot\b|\bisnt\b|\bdoesnt\b|\bno\b', lower_phrase):
#                     cat_name = None
#                 else:
#                     # cat_name = match_category(phrase_clean, method="substring")  # or method="substring" "fuzzy"
#                     cat_name = match_category_semantic(sub_phrase, categories)

#                 time_stamped_text.append({
#                     "begin_time": round(start_time, 2),
#                     "end_time": round(end_time, 2),
#                     "phrase": sub_phrase,
#                     "image_id": None,
#                     "category_name": cat_name or None,
#                     "bbox": None,
#                     "segmentation": None
#                 })

#                 phrase = ""
#                 start_time = None

#         # time_stamped_text.append({
#         #     "begin_time": round(seg["start"], 2),
#         #     "end_time": round(seg["end"], 2),
#         #     "phrase": phrase,
#         #     "image_id": None,
#         #     "category_name": cat_name,
#         #     "bbox": None,
#         #     "segmentation": None
#         # })

#     data = {
#         "full_text": full_text,
#         "time_stamped_text": time_stamped_text,
#         "categories": categories,
#     }

#     with open(output_json_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=4)
#     print(f"Saved full transcription to {output_json_path}")


def process_all_videos(root_input_dir, root_output_dir, interval_sec=10, overlap_sec=0):
    for dirpath, dirnames, filenames in os.walk(root_input_dir):
        dirnames.sort()
        filenames.sort()
        for file in filenames:
            if file.lower().endswith(".mp4") and not file.startswith("._"):
                video_path = os.path.join(dirpath, file)
                cam_subpath = os.path.relpath(dirpath, root_input_dir)
                video_name = os.path.splitext(file)[0]

                base_output = os.path.join(root_output_dir, cam_subpath, video_name)
                wav_dir = os.path.join(base_output, "audio_clips_wav")
                mp3_dir = os.path.join(base_output, "audio_clips_mp3")
                output_csv = os.path.join(base_output, "audio_labels.csv")
                output_json = os.path.join(base_output, "full_transcription.json")

                print(f"\nProcessing {video_path}")
                # extract_audio_segments_ffmpeg(video_path, output_csv, wav_dir, mp3_dir, interval_sec, overlap_sec)
                # save_full_audio(video_path, wav_dir, mp3_dir)
                transcribe_video_to_json_directly(video_path, output_json, all_speakers=True)
                # transcribe_full_video_to_json(video_path, output_json, all_speakers=True)  ### this one gives wrong time alignment between the phrases and the video
                # break  # Remove this line to process all videos in the directory
        # break # Remove this line to process all subdirectories recursively


# Example usage
if __name__ == "__main__":
    process_all_videos(
        root_input_dir=r"D:/trash_dataset/Brookhaven Landfill 20250602",
        root_output_dir=r"D:/trash_dataset/output/Brookhaven Landfill 20250602",
        interval_sec=10,  # Interval in seconds
        overlap_sec=0     # Overlap in seconds
    )
    
    process_all_videos(
        root_input_dir=r"D:/trash_dataset/Brookhaven MRF 20250604",
        root_output_dir=r"D:/trash_dataset/output/Brookhaven MRF 20250604",
        interval_sec=10,  # Interval in seconds
        overlap_sec=0     # Overlap in seconds
    )
    
    process_all_videos(
        root_input_dir=r"D:/trash_dataset/Brookhaven MRF 20250611",
        root_output_dir=r"D:/trash_dataset/output/Brookhaven MRF 20250611",
        interval_sec=10,  # Interval in seconds
        overlap_sec=0     # Overlap in seconds
    )

    process_all_videos(
        root_input_dir=r"D:/trash_dataset/Southold Dropoff 20250527",
        root_output_dir=r"D:/trash_dataset/output/Southold Dropoff 20250527",
        interval_sec=10,  # Interval in seconds
        overlap_sec=0     # Overlap in seconds
    )
    
    process_all_videos(
        root_input_dir=r"D:/trash_dataset/Southold Dropoff 20250528",
        root_output_dir=r"D:/trash_dataset/output/Southold Dropoff 20250528",
        interval_sec=10,  # Interval in seconds
        overlap_sec=0     # Overlap in seconds
    )
    
    
    # video_path = r"D:\\trash_dataset\\Brookhaven MRF 20250604_Output_Camera1_all\\first_5_min.mp4"
    # output_csv = r"D:/trash_dataset/output/Brookhaven MRF 20250604/Camera1/audio_labels.csv"
    # wav_dir = r"D:/trash_dataset/output/Brookhaven MRF 20250604/Camera1/audio_clips_wav"
    # mp3_dir = r"D:/trash_dataset/output/Brookhaven MRF 20250604/Camera1/audio_clips_mp3"
    # output_json = r"D:/trash_dataset/output/Brookhaven MRF 20250604/Camera1/full_transcription.json"

    # extract_audio_segments_ffmpeg(
    #     video_path=video_path,
    #     output_csv=output_csv,
    #     wav_dir=wav_dir,
    #     mp3_dir=mp3_dir,
    #     interval_sec=10,
    #     overlap_sec=0
    # )

    # save_full_audio(
    #     video_path=video_path,
    #     output_dir_wav=wav_dir,
    #     output_dir_mp3=mp3_dir
    # )

    # transcribe_full_video_to_json(
    #     video_path=video_path,
    #     output_json_path=output_json
    # )
    
    # extract_audio_segments_ffmpeg(
    #     video_path=r"D:\\trash_dataset\\output\\Brookhaven MRF 20250604\\Camera1\\first_5_min.mp4",
    #     output_csv=r"D:\\trash_dataset\\output\\Brookhaven MRF 20250604\\Camera1\\audio_labels.csv",
    #     interval_sec=10,  # ✅ supports float intervals
    #     overlap_sec=0    # ✅ supports float overlaps
    # )
