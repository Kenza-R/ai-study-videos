"""Caption generation module for video clips."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class WordGroup:
    """A group of words with timing information."""
    text: str
    start_time: float
    end_time: float


@dataclass
class CaptionConfig:
    """Configuration for caption styling."""
    font_name: str = "Arial"
    font_size: int = 100
    primary_color: str = "&H00FFFFFF"  # White
    outline_color: str = "&H00000000"  # Black
    background_color: str = "&H96000000"  # Semi-transparent black (~40% opacity)
    outline_width: int = 3
    bold: bool = True
    alignment: int = 5  # Center alignment (numpad position)
    margin_v: int = 80  # Vertical margin from bottom


def distribute_words_evenly(
    text: str,
    start_time: float,
    end_time: float,
    max_words: int = 2
) -> List[WordGroup]:
    """
    Distribute words evenly across the time duration.

    Args:
        text: The text to split into words
        start_time: Start time in seconds
        end_time: End time in seconds
        max_words: Maximum words per group (default: 2 for TikTok style)

    Returns:
        List of WordGroup objects with timing information
    """
    words = text.split()

    if not words:
        return []

    duration = end_time - start_time
    time_per_word = duration / len(words)

    word_groups = []
    for i in range(0, len(words), max_words):
        group_words = words[i:i + max_words]
        group_start = start_time + (i * time_per_word)
        group_end = group_start + (len(group_words) * time_per_word)

        word_groups.append(WordGroup(
            text=' '.join(group_words),
            start_time=group_start,
            end_time=group_end
        ))

    return word_groups


def format_ass_time(seconds: float) -> str:
    """
    Format time in seconds to ASS subtitle format (H:MM:SS.CC).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int((seconds % 1) * 100)

    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def escape_ass_text(text: str) -> str:
    """
    Escape special characters for ASS subtitle format.

    Args:
        text: Text to escape

    Returns:
        Escaped text
    """
    # ASS uses backslash for escaping
    text = text.replace('\\', '\\\\')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    return text


def generate_ass_file(
    word_groups: List[WordGroup],
    output_path: Path,
    config: CaptionConfig = CaptionConfig()
) -> None:
    """
    Generate an ASS subtitle file from word groups.

    Args:
        word_groups: List of WordGroup objects
        output_path: Path to save the ASS file
        config: Caption configuration
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ASS file header
    ass_content = [
        "[Script Info]",
        "Title: TikTok Style Captions",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{config.font_name},{config.font_size},{config.primary_color},&H000000FF,{config.outline_color},{config.background_color},{'-1' if config.bold else '0'},0,0,0,100,100,0,0,3,{config.outline_width},0,{config.alignment},10,10,{config.margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    # Add dialogue events for each word group
    for group in word_groups:
        start = format_ass_time(group.start_time)
        end = format_ass_time(group.end_time)
        text = escape_ass_text(group.text)

        dialogue = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        ass_content.append(dialogue)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ass_content))

    logger.info(f"Generated ASS file: {output_path}")


def burn_captions_to_video(
    video_path: Path,
    ass_path: Path,
    output_path: Path
) -> None:
    """
    Burn ASS captions into video using FFmpeg.

    Args:
        video_path: Path to input video
        ass_path: Path to ASS subtitle file
        output_path: Path to save output video

    Raises:
        subprocess.CalledProcessError: If FFmpeg command fails
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg command to burn subtitles
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f"ass={ass_path}",
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-c:a', 'copy',
        '-y',  # Overwrite output file
        str(output_path)
    ]

    logger.info(f"Burning captions into {video_path.name}...")
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ Created captioned video: {output_path}")

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        raise


def add_captions_to_scene(
    scene_data: dict,
    clips_dir: Path,
    output_dir: Path,
    max_words: int = 2,
    config: CaptionConfig = CaptionConfig()
) -> Path:
    """
    Add captions to a single scene video.

    Args:
        scene_data: Scene data from audio_metadata.json
        clips_dir: Directory containing scene videos
        output_dir: Directory to save captioned videos
        max_words: Maximum words per caption group
        config: Caption configuration

    Returns:
        Path to captioned video file
    """
    scene_idx = scene_data['scene_index']
    text = scene_data['text']
    start_time = 0.0  # Scene videos start at 0
    end_time = scene_data['clip_duration']

    # Verify video file exists
    video_path = clips_dir / f"scene_{scene_idx:02d}.mp4"
    if not video_path.exists():
        logger.info(f"Scene {scene_idx}: Skipping (no video file)")
        return None

    # Generate word groups
    word_groups = distribute_words_evenly(text, start_time, end_time, max_words)

    if not word_groups:
        logger.warning(f"Scene {scene_idx}: No words to caption")
        return None

    # Generate ASS file
    ass_path = output_dir / f"scene_{scene_idx:02d}.ass"
    generate_ass_file(word_groups, ass_path, config)

    # Burn captions into video
    output_path = output_dir / f"scene_{scene_idx:02d}_captioned.mp4"
    burn_captions_to_video(video_path, ass_path, output_path)

    return output_path


def add_captions_to_all_scenes(
    metadata_path: Path,
    clips_dir: Path = None,
    output_dir: Path = None,
    max_words: int = 2,
    font_size: int = 24,
    merge: bool = True
) -> List[Path]:
    """
    Add captions to all scene videos based on metadata.

    Args:
        metadata_path: Path to audio_metadata.json
        clips_dir: Directory containing scene videos (default: same dir as metadata + /clips)
        output_dir: Directory to save captioned videos (default: same dir as metadata + /clips_captioned)
        max_words: Maximum words per caption group
        font_size: Font size for captions
        merge: If True, merge all captioned videos into a single final video (default: True)

    Returns:
        List of paths to captioned videos
    """
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Set default directories
    if clips_dir is None:
        clips_dir = metadata_path.parent / "clips"
    else:
        clips_dir = Path(clips_dir)

    if output_dir is None:
        output_dir = metadata_path.parent / "clips_captioned"
    else:
        output_dir = Path(output_dir)

    # Create caption config
    config = CaptionConfig(font_size=font_size)

    # Process each scene
    captioned_videos = []
    scenes = metadata.get('scene_boundaries', [])

    logger.info(f"Adding captions to {len(scenes)} scenes...")
    logger.info(f"Settings: max_words={max_words}, font_size={font_size}")

    for scene_data in scenes:
        try:
            output_path = add_captions_to_scene(
                scene_data,
                clips_dir,
                output_dir,
                max_words,
                config
            )
            if output_path:
                captioned_videos.append(output_path)

        except Exception as e:
            scene_idx = scene_data.get('scene_index', '?')
            logger.error(f"Scene {scene_idx}: Failed to add captions: {e}")
            raise

    logger.info(f"✓ Successfully captioned {len(captioned_videos)} videos")
    logger.info(f"Output directory: {output_dir}")

    # Create marker file after all captions added successfully
    marker_path = output_dir / ".captions_complete"
    marker_path.touch()
    logger.info(f"Created completion marker: {marker_path}")

    # Optionally merge all captioned videos into a final video
    if merge and captioned_videos:
        # Import here to avoid circular dependency
        from video import concatenate_videos

        final_video_path = metadata_path.parent / "final_video.mp4"

        # Get the audio file path (should be in same dir as metadata)
        audio_path = metadata_path.parent / "audio.wav"
        if not audio_path.exists():
            logger.warning(f"Audio file not found at {audio_path}, using video audio")
            audio_path = None

        # Concatenate all captioned videos with our generated audio
        logger.info(f"Merging {len(captioned_videos)} captioned videos into final video...")
        concatenate_videos(captioned_videos, final_video_path, audio_path=audio_path)
        logger.info(f"✓ Final merged video saved: {final_video_path}")

    return captioned_videos
