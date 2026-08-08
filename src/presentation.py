"""
presentation.py — офлайн-згладжування треків і рендер демо-відео.

ЧОМУ ЦЕ ОКРЕМИЙ МОДУЛЬ ВІД inference.py
---------------------------------------
Продакшн-шлях причинний: на кадрі t відомі лише кадри <= t. Будь-яке
згладжування там — це фільтр по минулому, а він завжди дає фазову
затримку: рамка тягнеться за об'єктом.

Рендер презентації офлайн. Трек відомий цілком, тому можна:
  * згладжувати ВПЕРЕД І НАЗАД -> нульова затримка, рамка сидить на об'єкті;
  * інтерполювати пропуски -> знаючи, що об'єкт з'явився знову;
  * викидати треки заднім числом -> яких не було видно в реальному часі.

Жоден із цих трьох прийомів у realtime неможливий. Тому демо-відео
законно виглядає краще за живий вихід — але це не та сама система,
і на слайді це варто зазначити.

ВАЖЛИВО ПРО ЧЕСНІСТЬ
--------------------
`interpolate_gaps` домальовує рамку там, де детекції НЕ БУЛО.
Для показу траєкторії це нормально, для звіту про якість — ні.
Метрики рахуйте на сирому `predictions`, не на згладженому.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ================================================================
# 1. Перегрупування треків
# ================================================================

def regroup_tracks(tracks_by_frame: Dict) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """{frame: [{track_id, bbox}]} -> {track_id: [(frame, bbox), ...]} за часом."""
    grouped: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for frame_number, records in tracks_by_frame.items():
        for record in records:
            grouped[int(record["track_id"])].append(
                (int(frame_number), np.asarray(record["bbox"], dtype=np.float64))
            )
    for track_id in grouped:
        grouped[track_id].sort(key=lambda item: item[0])
    return dict(grouped)


# ================================================================
# 2. Згладжування
# ================================================================

def _ema_zero_phase(values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Експоненційне згладжування вперед, потім назад.

    Один прохід дає фазову затримку. Другий прохід у зворотному напрямку
    її точно компенсує (аналог scipy.signal.filtfilt, але без залежності
    від scipy). Результат: згладжено, зсуву немає.

    Ефективна ширина вікна подвоюється, тому alpha беріть удвічі більшу,
    ніж для звичайного EMA.
    """
    if len(values) < 2:
        return values.astype(np.float64, copy=True)
    alpha = float(np.clip(alpha, 1e-3, 1.0))

    forward = np.empty_like(values, dtype=np.float64)
    accumulator = float(values[0])
    for index, value in enumerate(values):
        accumulator = alpha * float(value) + (1.0 - alpha) * accumulator
        forward[index] = accumulator

    backward = np.empty_like(forward)
    accumulator = float(forward[-1])
    for index in range(len(forward) - 1, -1, -1):
        accumulator = alpha * float(forward[index]) + (1.0 - alpha) * accumulator
        backward[index] = accumulator
    return backward


def _interpolate_gaps(frames: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Заповнює пропущені кадри всередині треку лінійною інтерполяцією."""
    full_frames = np.arange(int(frames[0]), int(frames[-1]) + 1)
    filled = np.empty((len(full_frames), boxes.shape[1]), dtype=np.float64)
    for column in range(boxes.shape[1]):
        filled[:, column] = np.interp(full_frames, frames, boxes[:, column])
    observed = np.isin(full_frames, frames)
    return full_frames, filled, observed


def smooth_tracks(
    tracks_by_frame: Dict,
    alpha: float = 0.35,
    min_track_length: int = 6,
    interpolate_gaps: bool = True,
    max_gap: int = 12,
    stabilize_size: bool = True,
    size_percentile: float = 75.0,
) -> Dict[int, Dict]:
    """
    Перетворює сирі покадрові детекції на плавні траєкторії.

    Кроки, у порядку виконання:
      1. викинути треки, коротші за `min_track_length` (ретроспективний фільтр);
      2. заповнити пропуски до `max_gap` кадрів;
      3. перейти в (cx, cy, w, h) — згладжувати кути напряму погано, бо
         тремтіння позиції змішується з тремтінням розміру;
      4. згладити кожну з чотирьох величин фільтром з нульовою фазою;
      5. за бажанням зафіксувати розмір: для твердого об'єкта на сталій
         дистанції w і h стрибати не повинні, і саме ці стрибки найбільше
         впадають в око на відео.

    Повертає {track_id: {frames, boxes, observed}}, boxes у (x1,y1,x2,y2) float.
    """
    grouped = regroup_tracks(tracks_by_frame)
    smoothed: Dict[int, Dict] = {}

    for track_id, entries in grouped.items():
        if len(entries) < int(min_track_length):
            continue

        frames = np.array([frame for frame, _ in entries], dtype=np.int64)
        boxes = np.stack([box for _, box in entries]).astype(np.float64)

        if interpolate_gaps and len(frames) > 1:
            gaps = np.diff(frames)
            if np.any(gaps > int(max_gap)):
                # Великий розрив — це майже напевно два різні проходи об'єкта,
                # з'єднувати їх прямою було б вигадкою. Ріжемо на сегменти.
                split_points = np.where(gaps > int(max_gap))[0] + 1
                segments = np.split(np.arange(len(frames)), split_points)
            else:
                segments = [np.arange(len(frames))]
        else:
            segments = [np.arange(len(frames))]

        pieces = []
        for segment in segments:
            if len(segment) < int(min_track_length):
                continue
            segment_frames = frames[segment]
            segment_boxes = boxes[segment]
            if interpolate_gaps:
                segment_frames, segment_boxes, observed = _interpolate_gaps(
                    segment_frames, segment_boxes
                )
            else:
                observed = np.ones(len(segment_frames), dtype=bool)

            centre_x = (segment_boxes[:, 0] + segment_boxes[:, 2]) / 2.0
            centre_y = (segment_boxes[:, 1] + segment_boxes[:, 3]) / 2.0
            width = segment_boxes[:, 2] - segment_boxes[:, 0]
            height = segment_boxes[:, 3] - segment_boxes[:, 1]

            centre_x = _ema_zero_phase(centre_x, alpha)
            centre_y = _ema_zero_phase(centre_y, alpha)
            if stabilize_size:
                # Один розмір на весь трек: прибирає «дихання» рамки,
                # яке на відео помітніше за похибку позиції.
                width = np.full_like(width, float(np.percentile(width, size_percentile)))
                height = np.full_like(height, float(np.percentile(height, size_percentile)))
            else:
                width = _ema_zero_phase(width, alpha)
                height = _ema_zero_phase(height, alpha)

            result = np.stack([
                centre_x - width / 2.0, centre_y - height / 2.0,
                centre_x + width / 2.0, centre_y + height / 2.0,
            ], axis=1)
            pieces.append((segment_frames, result, observed))

        if not pieces:
            continue
        smoothed[track_id] = {
            "frames": np.concatenate([p[0] for p in pieces]),
            "boxes": np.concatenate([p[1] for p in pieces]),
            "observed": np.concatenate([p[2] for p in pieces]),
        }
    return smoothed


def smoothed_to_frame_index(smoothed: Dict[int, Dict]) -> Dict[int, List[Dict]]:
    """{track_id: {...}} -> {frame: [{track_id, bbox, observed, age, remaining}]}."""
    per_frame: Dict[int, List[Dict]] = defaultdict(list)
    for track_id, data in smoothed.items():
        total = len(data["frames"])
        for index, frame_number in enumerate(data["frames"]):
            per_frame[int(frame_number)].append({
                "track_id": int(track_id),
                "bbox": data["boxes"][index],
                "observed": bool(data["observed"][index]),
                "age": index,
                "remaining": total - index - 1,
            })
    return dict(per_frame)


# ================================================================
# 3. Рендер
# ================================================================

def _fade_alpha(age: int, remaining: int, fade_frames: int) -> float:
    """Плавна поява і зникнення: рамка не «вистрілює» і не обривається."""
    if fade_frames <= 0:
        return 1.0
    return float(min(1.0, (age + 1) / fade_frames, (remaining + 1) / fade_frames))


def _draw_corner_brackets(canvas, box, colour, thickness, gap_ratio=0.28, shift=3):
    """
    Кутові дужки замість суцільного прямокутника.

    Для об'єкта на 3-4 пікселі суцільна рамка фізично закриває те, що
    показує. Дужки лишають центр відкритим — глядач бачить сам об'єкт,
    а не тільки позначку навколо нього.

    `shift` вмикає субпіксельні координати OpenCV (1/8 пікселя при shift=3):
    без цього рамка сіпається на цілий піксель, і все згладжування
    втрачає сенс на етапі малювання.
    """
    factor = 1 << shift
    x1, y1, x2, y2 = [float(v) for v in box]
    width, height = x2 - x1, y2 - y1
    arm_x = max(2.0, width * gap_ratio)
    arm_y = max(2.0, height * gap_ratio)

    def point(x, y):
        return (int(round(x * factor)), int(round(y * factor)))

    segments = [
        ((x1, y1), (x1 + arm_x, y1)), ((x1, y1), (x1, y1 + arm_y)),
        ((x2, y1), (x2 - arm_x, y1)), ((x2, y1), (x2, y1 + arm_y)),
        ((x1, y2), (x1 + arm_x, y2)), ((x1, y2), (x1, y2 - arm_y)),
        ((x2, y2), (x2 - arm_x, y2)), ((x2, y2), (x2, y2 - arm_y)),
    ]
    for start, end in segments:
        cv2.line(canvas, point(*start), point(*end), colour,
                 thickness, cv2.LINE_AA, shift)


def render_presentation_video(
    input_path, output_path, smoothed: Dict[int, Dict],
    output_size=(1280, 720), colour=(80, 235, 120), thickness=2,
    box_margin: float = 6.0, fade_frames: int = 6, trail_length: int = 24,
    draw_trail: bool = True, draw_ids: bool = False, fps: Optional[float] = None,
    dim_unobserved: bool = True,
) -> dict:
    """
    Малює згладжені треки поверх відео.

    `box_margin` розширює рамку назовні: вона має обрамляти об'єкт із
    зазором, а не лежати на ньому. Для дальніх цілей це критично.

    `dim_unobserved` притлумлює кадри, де рамка інтерпольована, а не
    задетектована. Візуально чесніше: глядач бачить, де були дані.
    """
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Не вдалося відкрити відео: {input_path}")

    source_fps = fps or capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(output_size[0]), int(output_size[1])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
        float(source_fps), (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Не вдалося створити відео: {output_path}")

    per_frame = smoothed_to_frame_index(smoothed)
    trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=int(trail_length)))

    frame_number = 0
    drawn = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            entries = per_frame.get(frame_number, [])
            active_ids = {entry["track_id"] for entry in entries}
            for track_id in list(trails):
                if track_id not in active_ids:
                    trails.pop(track_id, None)

            if entries:
                overlay = frame.copy()
                for entry in entries:
                    x1, y1, x2, y2 = entry["bbox"]
                    box = (
                        x1 - box_margin, y1 - box_margin,
                        x2 + box_margin, y2 + box_margin,
                    )
                    centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    trails[entry["track_id"]].append(centre)

                    if draw_trail and len(trails[entry["track_id"]]) > 1:
                        points = list(trails[entry["track_id"]])
                        for index in range(1, len(points)):
                            weight = index / len(points)
                            trail_colour = tuple(int(c * weight) for c in colour)
                            cv2.line(
                                overlay,
                                (int(round(points[index - 1][0] * 8)), int(round(points[index - 1][1] * 8))),
                                (int(round(points[index][0] * 8)), int(round(points[index][1] * 8))),
                                trail_colour, 1, cv2.LINE_AA, 3,
                            )

                    entry_colour = colour
                    if dim_unobserved and not entry["observed"]:
                        entry_colour = tuple(int(c * 0.45) for c in colour)
                    _draw_corner_brackets(overlay, box, entry_colour, thickness)

                    if draw_ids:
                        cv2.putText(
                            overlay, f"#{entry['track_id']}",
                            (int(box[0]), max(12, int(box[1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, entry_colour, 1, cv2.LINE_AA,
                        )
                    drawn += 1

                alpha = max(
                    _fade_alpha(e["age"], e["remaining"], fade_frames) for e in entries
                )
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)

            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    return {
        "output_path": str(output_path),
        "frames_written": frame_number,
        "boxes_drawn": drawn,
        "tracks": len(smoothed),
        "fps": float(source_fps),
    }


def render_evidence_video(
    input_path, output_path, compensation_config=None, max_width: int = 640,
    step: int = 1, panel_width: int = 640, amplify: float = 6.0,
    fps: Optional[float] = None, max_frames: Optional[int] = None,
) -> dict:
    """
    Три панелі: оригінал | різниця БЕЗ компенсації | різниця З компенсацією.

    Це найпереконливіший кадр для слайда. Середня панель світиться вся —
    бо рухається камера. Права майже чорна, і на ній лишається тільки те,
    що рухається незалежно. Одна картинка пояснює, навіщо потрібен увесь
    класичний блок, без жодної формули.
    """
    import inference as inf

    config = dict(inf.PRODUCTION_CONFIG)
    if compensation_config is not None:
        config.update(dict(compensation_config))
    config["max_width"] = int(max_width)

    pipeline = inf.CompensationPipeline(config, step=int(step))
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Не вдалося відкрити відео: {input_path}")
    source_fps = fps or capture.get(cv2.CAP_PROP_FPS) or 30.0

    history: deque = deque(maxlen=int(step) + 1)
    writer = None
    written = 0
    seen = 0

    def label(panel, text):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(panel, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    try:
        while True:
            if max_frames is not None and seen >= int(max_frames):
                break
            ok, frame = capture.read()
            if not ok:
                break
            seen += 1

            gray, _ = inf.preprocess_frame(frame, int(max_width))
            sample = pipeline.process_frame(frame)
            history.append(gray)
            if not sample.valid or len(history) <= int(step):
                continue

            tensor = np.asarray(sample.tensor)
            current = tensor[2]
            warped_previous = tensor[1]
            validity = tensor[3]

            raw_difference = cv2.absdiff(gray, history[0]).astype(np.float32) / 255.0
            compensated = np.abs(current - warped_previous) * validity

            def to_panel(single_channel, colour_map=True):
                scaled = np.clip(single_channel * float(amplify), 0.0, 1.0)
                image = (scaled * 255).astype(np.uint8)
                if colour_map:
                    image = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                return image

            height = int(round(panel_width * gray.shape[0] / gray.shape[1]))
            size = (int(panel_width), height)
            panels = [
                label(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), "1. Вхідний кадр"),
                label(cv2.resize(to_panel(raw_difference), size), "2. Різниця БЕЗ компенсації"),
                label(cv2.resize(to_panel(compensated), size), "3. Різниця З компенсацією"),
            ]
            canvas = np.hstack(panels)

            if writer is None:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    float(source_fps), (canvas.shape[1], canvas.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Не вдалося створити відео: {output_path}")
            writer.write(canvas)
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return {"output_path": str(output_path), "frames_written": written}


def render_zoom_inset(
    frame, box, inset_size: int = 220, magnification: int = 6,
    corner: str = "top-right", colour=(80, 235, 120), margin: int = 16,
):
    """
    Врізка зі збільшенням цілі.

    Головний аргумент проєкту — дальність, а дальня ціль на слайді займає
    3 пікселі й у залі її просто не видно. Врізка 1:6 робить аргумент
    видимим. Для статичного слайда цінніша за саме відео.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    centre_x, centre_y = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(8, inset_size // (2 * magnification))

    sx1, sy1 = max(0, centre_x - half), max(0, centre_y - half)
    sx2, sy2 = min(width, centre_x + half), min(height, centre_y + half)
    patch = frame[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return frame

    patch = cv2.resize(patch, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(patch, (0, 0), (inset_size - 1, inset_size - 1), colour, 2)

    if corner == "top-right":
        ox, oy = width - inset_size - margin, margin
    elif corner == "top-left":
        ox, oy = margin, margin
    elif corner == "bottom-right":
        ox, oy = width - inset_size - margin, height - inset_size - margin
    else:
        ox, oy = margin, height - inset_size - margin

    frame[oy:oy + inset_size, ox:ox + inset_size] = patch
    cv2.line(frame, (centre_x, centre_y), (ox, oy + inset_size // 2),
             colour, 1, cv2.LINE_AA)
    return frame


# ================================================================
# 4. Кодек під PowerPoint / Keynote
# ================================================================

def transcode_for_slides(input_path, output_path, crf: int = 18, fps: Optional[float] = None) -> dict:
    """
    Перекодовує у H.264 + yuv420p.

    OpenCV пише mp4v (MPEG-4 Part 2). PowerPoint, Keynote і більшість
    браузерів його або не програють, або показують чорний прямокутник —
    класичний спосіб зіпсувати демо просто на сцені.

    `-pix_fmt yuv420p` не менш важливий за сам кодек: без нього H.264
    виходить у yuv444, який QuickTime і PowerPoint теж не відкриють.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg не знайдено. У Colab: !apt-get -qq install -y ffmpeg"
        )
    command = [
        ffmpeg, "-y", "-i", str(input_path),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    if fps is not None:
        command += ["-r", str(float(fps))]
    command.append(str(output_path))

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg завершився з помилкою:\n{result.stderr[-2000:]}")
    return {"output_path": str(output_path), "codec": "h264/yuv420p"}


# ================================================================
# 5. CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Презентаційний рендер згладжених треків.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks-json", default=None,
                        help="JSON із tracks_by_frame (з predict_video_bboxes_pipelined).")
    parser.add_argument("--output", default="demo_raw.mp4")
    parser.add_argument("--transcode", default=None,
                        help="Шлях для H.264/yuv420p версії під слайди.")
    parser.add_argument("--evidence", default=None,
                        help="Шлях для тripанельного відео (компенсація до/після).")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--min-track-length", type=int, default=6)
    parser.add_argument("--max-gap", type=int, default=12)
    parser.add_argument("--fade-frames", type=int, default=6)
    parser.add_argument("--trail-length", type=int, default=24)
    parser.add_argument("--box-margin", type=float, default=6.0)
    parser.add_argument("--no-trail", action="store_true")
    parser.add_argument("--no-stabilize-size", action="store_true")
    parser.add_argument("--no-interpolate", action="store_true")
    parser.add_argument("--draw-ids", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if args.evidence:
        report = render_evidence_video(
            args.video, args.evidence, max_frames=args.max_frames,
        )
        print("Evidence:", json.dumps(report, ensure_ascii=False))

    if args.tracks_json:
        with open(args.tracks_json, encoding="utf-8") as handle:
            raw = json.load(handle)
        tracks_by_frame = {int(k): v for k, v in raw.items()}

        smoothed = smooth_tracks(
            tracks_by_frame, alpha=args.alpha,
            min_track_length=args.min_track_length, max_gap=args.max_gap,
            interpolate_gaps=not args.no_interpolate,
            stabilize_size=not args.no_stabilize_size,
        )
        print(f"Треків після фільтра: {len(smoothed)}")

        report = render_presentation_video(
            args.video, args.output, smoothed,
            box_margin=args.box_margin, fade_frames=args.fade_frames,
            trail_length=args.trail_length, draw_trail=not args.no_trail,
            draw_ids=args.draw_ids,
        )
        print("Render:", json.dumps(report, ensure_ascii=False))

        if args.transcode:
            print("Transcode:", json.dumps(
                transcode_for_slides(args.output, args.transcode), ensure_ascii=False))


if __name__ == "__main__":
    main()
