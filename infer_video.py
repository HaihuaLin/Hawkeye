import os
import sys
import argparse
import tempfile
import torch
import numpy as np
from tqdm import tqdm

from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, KeywordsStoppingCriteria

# Optional cv2 for video splitting
try:
    import cv2
except ImportError:
    cv2 = None

# Optional decord
try:
    import decord
    from decord import VideoReader, cpu
except ImportError:
    decord = None


ANOMALY_PROMPT = (
    "Please determine whether the emotional attributes of the video are negative or not. "
    "If negative, answer 1, else answer 0. "
    "The answer should just contain 0 or 1 without other contents."
)


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def get_video_info(video_path: str):
    """Retrieve FPS, total frame count, and duration in seconds."""
    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return fps, frame_count, duration
    elif decord is not None:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps() or 25.0
        frame_count = len(vr)
        duration = frame_count / fps if fps > 0 else 0
        return fps, frame_count, duration
    else:
        # Fallback estimation
        return 25.0, 0, 0.0


def extract_clip(video_path: str, start_sec: float, end_sec: float, output_clip_path: str):
    """Extract a sub-clip using OpenCV."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for slicing video. Please install opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_clip_path, fourcc, fps, (width, height))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()


def predict_clip(
    model,
    tokenizer,
    video_processor,
    clip_path: str,
    pose_feature=None,
    scene_feature=None,
    device='cuda'
):
    """Run Hawkeye forward on a single clip."""
    # 1. Process video frames via LanguageBind
    video_tensor = video_processor(clip_path, return_tensors='pt')['pixel_values']
    if isinstance(video_tensor, list):
        tensor = [v.to(device, dtype=torch.float16) for v in video_tensor]
    else:
        tensor = video_tensor.to(device, dtype=torch.float16)

    # 2. Process pose feature (pad to 5 if needed)
    if pose_feature is None:
        tensor_pose = torch.zeros((5, 17, 5), dtype=torch.float16, device=device)
    else:
        if pose_feature.size(0) < 5:
            pad = torch.zeros((5 - pose_feature.size(0), 17, 5), dtype=torch.float16)
            pose_feature = torch.cat([pose_feature, pad], dim=0)
        tensor_pose = pose_feature[:5].to(device, dtype=torch.float16)

    # 3. Process scene feature (pad to 5 if needed)
    if scene_feature is None:
        tensor_scene = torch.zeros((5, 353), dtype=torch.float16, device=device)
    else:
        if scene_feature.size(0) < 5:
            pad = torch.zeros((5 - scene_feature.size(0), 353), dtype=torch.float16)
            scene_feature = torch.cat([scene_feature, pad], dim=0)
        tensor_scene = scene_feature[:5].to(device, dtype=torch.float16)

    # 4. Construct prompt
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    prompt_text = DEFAULT_X_TOKEN['VIDEO'] + '\n' + ANOMALY_PROMPT
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_X_token(
        prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt'
    ).unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    key = ['video']
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[tensor, [tensor_pose], [tensor_scene], key],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=16,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return output


def main():
    parser = argparse.ArgumentParser(description="Test Hawkeye model on input video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--model_path", type=str, default="../Model Zoo", help="Path to Hawkeye LoRA checkpoint directory")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5", help="Path or HF ID for base vicuna-7b-v1.5")
    parser.add_argument("--mode", type=str, choices=["scan", "overall"], default="scan",
                        help="'scan' scans through video segments; 'overall' tests entire video at once")
    parser.add_argument("--segment_sec", type=float, default=2.0, help="Duration of each segment in seconds for scan mode")
    parser.add_argument("--stride_sec", type=float, default=2.0, help="Stride in seconds between consecutive segments")
    parser.add_argument("--max_segments", type=int, default=100, help="Maximum number of segments to test (to save time)")
    parser.add_argument("--load_4bit", action="store_true", help="Enable 4-bit quantization (if low GPU VRAM)")
    parser.add_argument("--load_8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print("=" * 60)
    print(" Hawkeye Anomaly Detection Inference")
    print(f" Target Video : {args.video}")
    print(f" Model Path   : {args.model_path}")
    print(f" Base Model   : {args.model_base}")
    print(f" Device       : {args.device}")
    print("=" * 60)

    disable_torch_init()

    # Ensure model_name contains 'lora' to trigger PeftModel loading in builder.py
    model_name = "hawkeye_lora"

    print("\n[1/3] Loading Hawkeye model and LanguageBind video tower...")
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device
    )
    video_processor = processor['video']
    print("Model loaded successfully!\n")

    fps, frame_count, duration = get_video_info(args.video)
    print(f"Video Info: FPS={fps:.2f}, Total Frames={frame_count}, Duration={duration:.2f}s ({format_timestamp(duration)})")

    print(f"\n[2/3] Running Inference (Mode: {args.mode})...")

    if args.mode == "overall" or duration <= args.segment_sec:
        # Evaluate whole video as a single sample
        res = predict_clip(model, tokenizer, video_processor, args.video, device=args.device)
        is_anomaly = "1" in res
        status = "ANOMALY DETECTED (1)" if is_anomaly else "NORMAL (0)"
        print(f"\nResult for entire video: {status} (Raw output: {res})")
        return

    # Scan mode: slice video into segments
    segments = []
    current_time = 0.0
    while current_time + 0.5 < duration and len(segments) < args.max_segments:
        end_time = min(current_time + args.segment_sec, duration)
        segments.append((current_time, end_time))
        current_time += args.stride_sec

    print(f"Split video into {len(segments)} segments (segment length={args.segment_sec}s, stride={args.stride_sec}s):")

    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, (st, et) in enumerate(tqdm(segments, desc="Testing segments")):
            clip_path = os.path.join(temp_dir, f"clip_{idx}.mp4")
            extract_clip(args.video, st, et, clip_path)

            pred = predict_clip(
                model=model,
                tokenizer=tokenizer,
                video_processor=video_processor,
                clip_path=clip_path,
                device=args.device
            )

            is_anomaly = "1" in pred
            results.append({
                "start": st,
                "end": et,
                "pred": pred,
                "is_anomaly": is_anomaly
            })

    print("\n[3/3] ===================== Detection Timeline =====================")
    anomalous_intervals = []
    current_interval = None

    for r in results:
        st_str = format_timestamp(r['start'])
        et_str = format_timestamp(r['end'])
        tag = "[! ANOMALOUS (1) !]" if r['is_anomaly'] else "[  Normal (0)  ]"
        print(f"[{st_str} - {et_str}]  {tag}  (output: {r['pred']})")

        if r['is_anomaly']:
            if current_interval is None:
                current_interval = [r['start'], r['end']]
            else:
                current_interval[1] = r['end']
        else:
            if current_interval is not None:
                anomalous_intervals.append(current_interval)
                current_interval = None

    if current_interval is not None:
        anomalous_intervals.append(current_interval)

    total = len(results)
    anomaly_count = sum(1 for r in results if r['is_anomaly'])
    print("\n" + "=" * 60)
    print(" SUMMARY REPORT:")
    print(f" - Total segments evaluated : {total}")
    print(f" - Anomalous segments count : {anomaly_count} ({anomaly_count / total * 100:.1f}%)")

    if anomalous_intervals:
        print(" - Detected Anomaly Time Windows:")
        for st, et in anomalous_intervals:
            print(f"   * {format_timestamp(st)} --> {format_timestamp(et)}")
    else:
        print(" - No anomalous sentiment detected in this video.")
    print("=" * 60)


if __name__ == "__main__":
    main()
