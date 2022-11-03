import os
import json

import numpy as np
import ffmpeg
import whisper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.tree import DecisionTreeRegressor
import torch
import youtube_dl
import pandas as pd
import streamlit as st
import altair as alt

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

YDL_OPTS = {
    "download_archive": os.path.join(DATA_DIR, "archive.txt"),
    "format": "bestaudio/best",
    "outtmpl": os.path.join(DATA_DIR, "%(title)s.%(ext)s"),
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}

llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
device = "cuda" if torch.cuda.is_available() else "cpu"


def download(url, ydl_opts):
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info("{}".format(url))
        fname = ydl.prepare_filename(result)
        return fname


def transcribe(audio_path, transcript_path):
    if os.path.exists(transcript_path):
        with open(transcript_path, "r") as f:
            result = json.load(f)
    else:
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(audio_path)
        with open(transcript_path, "w") as f:
            json.dump(result, f)
    return result["segments"]


def compute_seg_durations(segments):
    return [s["end"] - s["start"] for s in segments]


def compute_info_densities(
    segments, seg_durations, llm, tokenizer, device, ctxt_len=512
):
    seg_encodings = [tokenizer(seg["text"], return_tensors="pt") for seg in segments]
    input_ids = [enc.input_ids.to(device) for enc in seg_encodings]
    seg_lens = [x.shape[1] for x in input_ids]
    cat_input_ids = torch.cat(input_ids, axis=1)
    end = 0
    seg_nlls = []
    n = cat_input_ids.shape[1]
    for i, seg_len in enumerate(seg_lens):
        end = min(n, end + seg_len)
        start = max(0, end - ctxt_len)
        ctxt_ids = cat_input_ids[:, start:end]
        target_ids = ctxt_ids.clone()
        target_ids[:, :-seg_len] = -100
        avg_nll = llm(ctxt_ids, labels=target_ids).loss.detach().numpy()
        nll = avg_nll * seg_len
        seg_nlls.append(nll)
    seg_nlls = np.array(seg_nlls)
    info_densities = seg_nlls / seg_durations
    return info_densities


def smooth_info_densities(info_densities, seg_durations, max_leaf_nodes, min_sec_leaf):
    min_samples_leaf = int(np.ceil(min_sec_leaf / np.mean(seg_durations)))
    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf
    )
    X = np.arange(0, len(info_densities), 1)[:, np.newaxis]
    tree.fit(X, info_densities)
    smoothed_info_densities = tree.predict(X)
    return smoothed_info_densities


def squash_segs(segments, info_densities):
    start = segments[0]["start"]
    end = None
    seg_times = []
    seg_densities = [info_densities[0]]
    for i in range(1, len(segments)):
        curr_density = info_densities[i]
        if curr_density != info_densities[i - 1]:
            seg = segments[i]
            seg_start = seg["start"]
            seg_times.append((start, seg_start))
            seg_densities.append(curr_density)
            start = seg_start
    seg_times.append((start, segments[-1]["end"]))
    return seg_times, seg_densities


def compute_speedups(info_densities):
    avg_density = np.mean(info_densities)
    speedups = avg_density / info_densities
    return speedups


def compute_actual_speedup(durations, speedups, total_duration):
    spedup_durations = durations / speedups
    spedup_total_duration = spedup_durations.sum()
    actual_speedup_factor = total_duration / spedup_total_duration
    return spedup_total_duration, actual_speedup_factor


def postprocess_speedups(
    speedups, factor, min_speedup, max_speedup, durations, total_duration, thresh=0.01
):
    assert min_speedup <= factor and factor <= max_speedup
    tuned_factor = np.array([factor / 10, factor * 10])
    actual_speedup_factor = None
    while (
        actual_speedup_factor is None
        or abs(actual_speedup_factor - factor) / factor > thresh
    ):
        mid = tuned_factor.mean()
        tuned_speedups = speedups * mid
        tuned_speedups = np.round(tuned_speedups, decimals=2)
        tuned_speedups = np.clip(tuned_speedups, min_speedup, max_speedup)
        _, actual_speedup_factor = compute_actual_speedup(
            durations, tuned_speedups, total_duration
        )
        tuned_factor[0 if actual_speedup_factor < factor else 1] = mid
    return tuned_speedups


def cat_clips(seg_times, speedups, audio_path, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    in_file = ffmpeg.input(audio_path)
    segs = []
    for (start, end), speedup in zip(seg_times, speedups):
        seg = in_file.filter("atrim", start=start, end=end).filter("atempo", speedup)
        segs.append(seg)
    cat = ffmpeg.concat(*segs, v=0, a=1)
    cat.output(output_path).run()


def format_duration(duration):
    s = duration % 60
    m = duration // 60
    h = m // 60
    return "%02d:%02d:%02d" % (h, m, s)


def strike(url, speedup_factor, min_speedup, max_speedup, max_num_segments):

    min_speedup = max(0.5, min_speedup)  # ffmpeg limit

    with st.spinner("downloading..."):
        name = download(url, YDL_OPTS)
    assert name.endswith(".m4a")
    name = name.split(".m4a")[0].split("/")[-1]

    audio_path = os.path.join(DATA_DIR, "%s.mp3" % name)
    transcript_path = os.path.join(DATA_DIR, "%s.json" % name)
    output_path = os.path.join(DATA_DIR, "%s_smooth.mp3" % name)

    with st.spinner("transcribing..."):
        segments = transcribe(audio_path, transcript_path)

    seg_durations = compute_seg_durations(segments)

    with st.spinner("calculating information density..."):
        info_densities = compute_info_densities(
            segments, seg_durations, llm, tokenizer, device
        )

    total_duration = segments[-1]["end"] - segments[0]["start"]
    min_sec_leaf = total_duration / max_num_segments
    smoothed_info_densities = smooth_info_densities(
        info_densities, seg_durations, max_num_segments, min_sec_leaf
    )

    squashed_times, squashed_densities = squash_segs(segments, smoothed_info_densities)
    squashed_durations = np.array([end - start for start, end in squashed_times])

    speedups = compute_speedups(squashed_densities)
    speedups = postprocess_speedups(
        speedups,
        speedup_factor,
        min_speedup,
        max_speedup,
        squashed_durations,
        total_duration,
    )

    with st.spinner("stitching segments..."):
        cat_clips(squashed_times, speedups, audio_path, output_path)

    spedup_total_duration, actual_speedup_factor = compute_actual_speedup(
        squashed_durations, speedups, total_duration
    )
    st.write("original duration: %s" % format_duration(total_duration))
    st.write("new duration: %s" % format_duration(spedup_total_duration))
    st.write("speedup: %0.2f" % actual_speedup_factor)

    times = np.array([(seg["start"] + seg["end"]) / 2 for seg in segments])
    times /= 60
    annotations = [seg["text"] for seg in segments]
    data = [times, info_densities / np.log(2), annotations]
    cols = ["time (minutes)", "bits per second", "transcript"]
    df = pd.DataFrame(list(zip(*data)), columns=cols)
    min_time = segments[0]["start"] / 60
    max_time = segments[-1]["end"] / 60
    lines = (
        alt.Chart(df, title="information rate")
        .mark_line(color="gray", opacity=0.5)
        .encode(
            x=alt.X(cols[0], scale=alt.Scale(domain=(min_time, max_time))),
            y=cols[1],
        )
    )
    dots = (
        alt.Chart(df)
        .mark_circle(size=50, opacity=1)
        .encode(
            x=alt.X(cols[0], scale=alt.Scale(domain=(min_time, max_time))),
            y=cols[1],
            tooltip=["transcript"],
        )
    )
    st.altair_chart((lines + dots).interactive(), use_container_width=True)

    times = sum([list(x) for x in squashed_times], [])
    times = np.array(times)
    times /= 60
    data = [times, np.repeat(speedups, 2)]
    cols = ["time (minutes)", "speedup"]
    df = pd.DataFrame(list(zip(*data)), columns=cols)
    min_actual_speedups = min(speedups)
    max_actual_speedups = max(speedups)
    eps = 0.1
    lines = (
        alt.Chart(df, title="speedup based on information rate")
        .mark_line()
        .encode(
            x=alt.X(cols[0], scale=alt.Scale(domain=(min_time, max_time))),
            y=alt.Y(
                cols[1],
                scale=alt.Scale(
                    domain=(min_actual_speedups - eps, max_actual_speedups + eps)
                ),
            ),
        )
    )
    st.altair_chart(lines.interactive(), use_container_width=True)

    return output_path


with st.form("my_form"):
    url = st.text_input(
        "youtube url", value="https://www.youtube.com/watch?v=_3MBQm7GFIM"
    )
    speedup_factor = st.slider("speedup", min_value=1.0, max_value=10.0, value=1.5)
    min_speedup = 1
    max_speedup = st.slider("maximum speedup", min_value=1.0, max_value=10.0, value=2.0)
    speedup_factor = min(speedup_factor, max_speedup)
    max_num_segments = st.slider(
        "variance in speedup over time", min_value=2, max_value=100, value=20
    )
    submitted = st.form_submit_button("submit")
    if submitted:
        st.write("original video:")
        st.video(url)
        output_path = strike(
            url, speedup_factor, min_speedup, max_speedup, max_num_segments
        )
        st.write("processed audio:")
        st.audio(output_path)
