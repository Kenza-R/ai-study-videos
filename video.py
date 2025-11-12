"""Video generation module for paper video clips."""

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from runwayml import RunwayML

from audio import load_audio_metadata, SceneAudio

logger = logging.getLogger(__name__)

# Video generation configuration
VIDEO_MODEL = 'veo3.1'
POLL_INTERVAL = 1  # seconds between status checks
MAX_WORKERS = 5  # Runway has better rate limits than Google Veo


@dataclass(frozen=True)
class VideoClip:
    """Information about a generated video clip."""
    scene_index: int
    clip_path: str
    duration: float
    prompt: str
    visual_type: str


@dataclass(frozen=True)
class VideoGenerationResult:
    """Complete video generation result."""
    clips: List[VideoClip]
    output_dir: str
    total_clips: int


def _generate_single_video(
    client: RunwayML,
    scene_audio: SceneAudio,
    output_dir: Path,
    poll_interval: int = POLL_INTERVAL
) -> VideoClip:
    """
    Generate a single video clip for a scene.

    Args:
        client: Runway API client
        scene_audio: SceneAudio object with scene information
        output_dir: Directory to save video clip
        poll_interval: Seconds between status checks

    Returns:
        VideoClip object with generation result

    Raises:
        Exception: If video generation fails
    """
    scene_idx = scene_audio.scene_index
    prompt = scene_audio.visual_content

    # Check if clip already exists
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_filename = f"scene_{scene_idx:02d}.mp4"
    clip_path = output_dir / clip_filename

    if clip_path.exists():
        logger.info(f"Scene {scene_idx}: Skipping - clip already exists at {clip_path}")
        return VideoClip(
            scene_index=scene_idx,
            clip_path=str(clip_path),
            duration=scene_audio.clip_duration,
            prompt=prompt,
            visual_type=scene_audio.visual_type
        )

    logger.info(f"Scene {scene_idx}: Starting video generation...")
    logger.debug(f"Scene {scene_idx} prompt: {prompt}")

    try:
        # Start video generation with TikTok vertical aspect ratio (1280:720 is 16:9, we want 9:16)
        # Runway uses ratio as "width:height"
        # Duration must match the audio clip duration for proper sync
        # Veo3.1 only supports duration values of 4, 6, or 8 seconds
        # Choose the closest allowed duration to the audio clip length
        allowed_durations = [4, 6, 8]
        video_duration = min(allowed_durations, key=lambda x: abs(x - scene_audio.clip_duration))

        task = client.text_to_video.create(
            model=VIDEO_MODEL,
            prompt_text=prompt,
            ratio="720:1280",  # Vertical video for TikTok/Instagram Reels (9:16)
            duration=video_duration,  # Closest allowed duration (4, 6, or 8 seconds)
        )
        logger.debug(f"Scene {scene_idx}: Requested video duration: {video_duration}s (audio: {scene_audio.clip_duration:.2f}s)")
        task_id = task.id
        logger.debug(f"Scene {scene_idx}: Task created with ID {task_id}")

        # Poll until complete
        time.sleep(poll_interval)
        task = client.tasks.retrieve(task_id)

        while task.status not in ['SUCCEEDED', 'FAILED']:
            time.sleep(poll_interval)
            task = client.tasks.retrieve(task_id)
            logger.debug(f"Scene {scene_idx}: Status - {task.status}")

        if task.status == 'FAILED':
            raise Exception(f"Task failed: {task}")

        # Download the video
        logger.info(f"Scene {scene_idx}: Video generation complete, downloading...")

        # The task should have the video URL in the output
        video_url = task.output[0] if isinstance(task.output, list) else task.output

        # Download video from URL
        import requests
        response = requests.get(video_url)
        response.raise_for_status()

        # Save to temporary file first
        temp_path = clip_path.with_suffix('.temp.mp4')
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        # Trim video to exact audio duration if needed
        # Veo3.1 only supports 4, 6, 8s durations, so we may need to trim
        if video_duration > scene_audio.clip_duration:
            logger.info(f"Scene {scene_idx}: Trimming video from {video_duration}s to {scene_audio.clip_duration:.2f}s")
            import subprocess

            # Use ffmpeg to trim video to exact audio length
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(temp_path),
                '-t', str(scene_audio.clip_duration),  # Trim to exact duration
                '-c', 'copy',  # Copy codec (fast, no re-encoding)
                str(clip_path)
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Scene {scene_idx}: FFmpeg trim failed, using full video: {result.stderr}")
                # Fall back to using the full video
                temp_path.rename(clip_path)
            else:
                # Remove temp file after successful trim
                temp_path.unlink()
                logger.debug(f"Scene {scene_idx}: Video trimmed successfully")
        else:
            # No trimming needed, just rename
            temp_path.rename(clip_path)

        logger.info(f"Scene {scene_idx}: Saved to {clip_path}")

        return VideoClip(
            scene_index=scene_idx,
            clip_path=str(clip_path),
            duration=scene_audio.clip_duration,
            prompt=prompt,
            visual_type=scene_audio.visual_type
        )

    except Exception as e:
        logger.error(f"Scene {scene_idx}: Video generation failed: {e}")
        raise Exception(f"Failed to generate video for scene {scene_idx}: {e}")


def concatenate_videos(
    video_paths: list[Path],
    output_path: Path,
    audio_path: Path | None = None,
) -> None:
    """
    Concatenate multiple video files into a single video.

    Uses ffmpeg concat demuxer for lossless concatenation.
    All input videos must have the same codec and resolution.

    IMPORTANT: The individual video clips may contain unwanted audio from the
    generation process. If audio_path is provided, the concatenated video will
    use that audio track instead of the audio from the individual clips.

    Args:
        video_paths: List of video file paths to concatenate (in order)
        output_path: Path where the concatenated video will be saved
        audio_path: Optional path to audio file to use instead of video audio

    Raises:
        ValueError: If video_paths is empty
        subprocess.CalledProcessError: If ffmpeg command fails
    """
    if not video_paths:
        raise ValueError("Cannot concatenate empty list of videos")

    # Create concat file listing all videos
    concat_file = output_path.parent / "concat.txt"

    try:
        with open(concat_file, "w") as f:
            for video_path in video_paths:
                # Use absolute paths and escape single quotes
                abs_path = video_path.resolve()
                f.write(f"file '{abs_path}'\n")


        # Build ffmpeg command
        # If we have a separate audio file, strip audio from videos and add our audio
        if audio_path:
            # First concatenate videos without audio, then add our audio track
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),  # Video input (concatenated videos)
                "-i", str(audio_path),    # Audio input (our generated TTS audio)
                "-map", "0:v",            # Use video from first input (includes burned captions)
                "-map", "1:a",            # Use audio from second input (our TTS audio)
                "-c:v", "copy",           # Copy video codec (no re-encoding, preserves burned captions)
                "-c:a", "aac",            # Encode audio to AAC
                "-shortest",              # Stop at shortest stream (in case audio/video mismatch)
                "-y",
                str(output_path),
            ]
            logger.info(f"Concatenating {len(video_paths)} videos with audio from: {audio_path}")
        else:
            # Original behavior: concatenate with existing audio
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                "-y",
                str(output_path),
            ]
            logger.info(f"Concatenating {len(video_paths)} videos into: {output_path}")

        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg concatenation failed: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

        logger.info(f"Concatenated video saved: {output_path}")

    finally:
        # Clean up concat file
        if concat_file.exists():
            concat_file.unlink()


def generate_videos(
    metadata_path: Path,
    output_dir: Path | None = None,
    api_key: str | None = None,
    max_workers: int = MAX_WORKERS,
    poll_interval: int = POLL_INTERVAL,
    merge: bool = False
) -> VideoGenerationResult:
    """
    Generate video clips for all scenes in parallel.

    Args:
        metadata_path: Path to audio_metadata.json file
        output_dir: Directory to save video clips (default: same as metadata_path)
        api_key: Runway API key (defaults to RUNWAYML_API_SECRET env var)
        max_workers: Maximum parallel video generations
        poll_interval: Seconds between status checks
        merge: If True, concatenate all generated clips into a single final video

    Returns:
        VideoGenerationResult with all generated clips

    Raises:
        ValueError: If API key is missing or metadata file is invalid
        FileNotFoundError: If metadata file doesn't exist
        Exception: If video generation fails
    """
    if api_key is None:
        api_key = os.getenv('RUNWAYML_API_SECRET')

    if not api_key:
        raise ValueError("RUNWAYML_API_SECRET environment variable not set")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load audio metadata
    logger.info(f"Loading metadata from {metadata_path}")
    audio_result = load_audio_metadata(metadata_path)

    # Set output directory
    if output_dir is None:
        output_dir = metadata_path.parent / "clips"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving clips to {output_dir}")

    # Initialize Runway client
    client = RunwayML(api_key=api_key)

    # Count how many videos we need to generate
    logger.info(f"Generating {len(audio_result.scene_boundaries)} video clips")

    # Generate videos in parallel
    clips = []

    def generate_scene(scene_audio: SceneAudio) -> VideoClip:
        """Wrapper for parallel execution."""
        return _generate_single_video(client, scene_audio, output_dir, poll_interval)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_scene = {
            executor.submit(generate_scene, scene_audio): scene_audio
            for scene_audio in audio_result.scene_boundaries
        }

        # Collect results as they complete
        for future in as_completed(future_to_scene):
            scene_audio = future_to_scene[future]
            try:
                clip = future.result()
                clips.append(clip)
                logger.info(f"✓ Scene {clip.scene_index} complete: {clip.clip_path}")
            except Exception as e:
                logger.error(f"✗ Scene {scene_audio.scene_index} failed: {e}")
                raise

    # Sort clips by scene index
    clips.sort(key=lambda c: c.scene_index)

    # Create marker file after all videos generated successfully
    marker_path = output_dir / ".videos_complete"
    marker_path.touch()
    logger.info(f"Created completion marker: {marker_path}")

    # Optionally merge all videos into a final video
    if merge:
        # Get list of video clip paths
        video_clip_paths = [Path(clip.clip_path) for clip in clips]

        if video_clip_paths:
            final_video_path = metadata_path.parent / "final_video.mp4"

            # Verify all clip files exist before concatenation
            missing_files = [vp for vp in video_clip_paths if not vp.exists()]
            if missing_files:
                logger.error(f"Cannot merge: missing video files: {missing_files}")
                raise FileNotFoundError(f"Missing video files: {missing_files}")

            # Get the audio file path (should be in same dir as metadata)
            audio_path = metadata_path.parent / "audio.wav"
            if not audio_path.exists():
                logger.warning(f"Audio file not found at {audio_path}, using video audio")
                audio_path = None

            # Concatenate all scene videos with our generated audio
            concatenate_videos(video_clip_paths, final_video_path, audio_path=audio_path)
            logger.info(f"Final merged video saved: {final_video_path}")
        else:
            logger.warning("No video clips to merge")

    return VideoGenerationResult(
        clips=clips,
        output_dir=str(output_dir),
        total_clips=len(clips)
    )


def save_video_metadata(result: VideoGenerationResult, output_path: Path) -> None:
    """
    Save video generation metadata to JSON file.

    Args:
        result: VideoGenerationResult object
        output_path: Path to output JSON file

    Raises:
        IOError: If file cannot be written
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            'output_dir': result.output_dir,
            'total_clips': result.total_clips,
            'clips': [
                {
                    'scene_index': clip.scene_index,
                    'clip_path': clip.clip_path,
                    'duration': clip.duration,
                    'prompt': clip.prompt,
                    'visual_type': clip.visual_type
                }
                for clip in result.clips
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved video metadata to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save video metadata: {e}")
        raise IOError(f"Could not write metadata to {output_path}: {e}")
