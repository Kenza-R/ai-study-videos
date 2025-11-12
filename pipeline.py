"""Pipeline orchestration for end-to-end video generation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from audio import generate_audio, save_audio_metadata
from captions import add_captions_to_all_scenes
from pubmed import fetch_paper
from scenes import generate_scenes, save_scenes, load_scenes
from video import generate_videos, save_video_metadata

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Represents a step in the video generation pipeline."""

    name: str
    description: str
    check_completion: Callable[[], bool]
    execute: Callable[[], None]


class PipelineError(Exception):
    """Raised when a pipeline step fails."""

    pass


def check_paper_fetched(output_dir: Path) -> bool:
    """Check if paper has been fetched."""
    paper_json = output_dir / "paper.json"
    return paper_json.exists()


def check_script_generated(output_dir: Path) -> bool:
    """Check if script has been generated."""
    script_json = output_dir / "script.json"
    return script_json.exists()


def check_audio_generated(output_dir: Path) -> bool:
    """Check if audio has been generated."""
    audio_file = output_dir / "audio.wav"
    metadata_file = output_dir / "audio_metadata.json"
    return audio_file.exists() and metadata_file.exists()


def check_videos_generated(output_dir: Path) -> bool:
    """Check if videos have been generated for all scenes."""
    clips_dir = output_dir / "clips"
    marker_path = clips_dir / ".videos_complete"
    return marker_path.exists()


def check_captions_added(output_dir: Path) -> bool:
    """Check if captions have been added to all videos."""
    clips_captioned_dir = output_dir / "clips_captioned"
    marker_path = clips_captioned_dir / ".captions_complete"
    return marker_path.exists()


def orchestrate_pipeline(
    pmid: str,
    output_dir: Path,
    skip_existing: bool = True,
    stop_after: Optional[str] = None,
    voice: str = "Kore",
    max_workers: int = 5,
    merge: bool = True,
) -> None:
    """Orchestrate the complete video generation pipeline.

    Args:
        pmid: PubMed ID or PMC ID of the paper
        output_dir: Directory for all output files
        skip_existing: If True, skip completed steps (idempotent)
        stop_after: Stop after this step name (for debugging)
        voice: Gemini TTS voice to use
        max_workers: Maximum parallel workers for video generation
        merge: If True, concatenate all video clips into a single final video (default: True)

    Raises:
        PipelineError: If any step fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define pipeline steps
    steps = [
        PipelineStep(
            name="fetch-paper",
            description=f"Fetching paper {pmid} from PubMed Central",
            check_completion=lambda: check_paper_fetched(output_dir),
            execute=lambda: fetch_paper(pmid, str(output_dir)),
        ),
        PipelineStep(
            name="generate-script",
            description="Generating video script with scenes",
            check_completion=lambda: check_script_generated(output_dir),
            execute=lambda: _generate_script_step(output_dir),
        ),
        PipelineStep(
            name="generate-audio",
            description="Generating audio for all scenes",
            check_completion=lambda: check_audio_generated(output_dir),
            execute=lambda: _generate_audio_step(output_dir, voice),
        ),
        PipelineStep(
            name="generate-videos",
            description="Generating videos for all scenes",
            check_completion=lambda: check_videos_generated(output_dir),
            execute=lambda: _generate_videos_step(output_dir, max_workers, merge),
        ),
        PipelineStep(
            name="add-captions",
            description="Adding captions to all videos",
            check_completion=lambda: check_captions_added(output_dir),
            execute=lambda: _add_captions_step(output_dir, merge=merge),
        ),
    ]

    logger.info(f"Starting pipeline for PMID {pmid}")
    logger.info(f"Output directory: {output_dir}")

    for step in steps:
        logger.info(f"Step: {step.name}")

        # Check if step is already complete
        if skip_existing and step.check_completion():
            logger.info(f"  ✓ Already complete, skipping")
            if stop_after == step.name:
                logger.info(f"Stopping after {step.name} as requested")
                break
            continue

        # Execute step
        logger.info(f"  → {step.description}")
        try:
            step.execute()
            logger.info(f"  ✓ Complete")
        except Exception as e:
            error_msg = f"Step '{step.name}' failed: {e}"
            logger.error(f"  ✗ {error_msg}")
            raise PipelineError(error_msg) from e

        # Stop if requested
        if stop_after == step.name:
            logger.info(f"Stopping after {step.name} as requested")
            break

    logger.info("Pipeline complete!")
    logger.info(f"Output files in: {output_dir}")


def _generate_script_step(output_dir: Path) -> None:
    """Execute the generate-script step."""
    # Load paper data
    paper_file = output_dir / "paper.json"
    with open(paper_file, "r", encoding="utf-8") as f:
        paper_data = json.load(f)

    # Generate scenes
    scene_list = generate_scenes(paper_data)

    # Save to script.json
    script_file = output_dir / "script.json"
    save_scenes(scene_list, script_file)

    logger.info(f"Generated {len(scene_list)} scenes")


def _generate_audio_step(output_dir: Path, voice: str) -> None:
    """Execute the generate-audio step."""
    # Load scenes
    script_file = output_dir / "script.json"
    scenes = load_scenes(script_file)

    # Generate audio
    result = generate_audio(scenes, output_dir, voice=voice)

    # Save metadata
    metadata_file = output_dir / "audio_metadata.json"
    save_audio_metadata(result, metadata_file)

    logger.info(f"Generated audio: {result.total_duration:.2f}s with voice '{voice}'")


def _generate_videos_step(output_dir: Path, max_workers: int, merge: bool = False) -> None:
    """Execute the generate-videos step.

    Note: The merge parameter is kept for API compatibility but merging now happens
    after captions are added in the _add_captions_step.
    """
    # Load audio metadata
    metadata_path = output_dir / "audio_metadata.json"

    # Generate videos (merge=False since we'll merge after captions)
    result = generate_videos(
        metadata_path, output_dir=None, max_workers=max_workers, poll_interval=1, merge=False
    )

    # Save video metadata
    video_metadata_file = Path(result.output_dir) / "video_metadata.json"
    save_video_metadata(result, video_metadata_file)

    logger.info(f"Generated {result.total_clips} video clips")


def _add_captions_step(output_dir: Path, merge: bool = True) -> None:
    """Execute the add-captions step."""
    # Load audio metadata
    metadata_path = output_dir / "audio_metadata.json"

    # Add captions to all scenes
    captioned_videos = add_captions_to_all_scenes(
        metadata_path, clips_dir=None, output_dir=None, max_words=2, font_size=24, merge=merge
    )

    logger.info(f"Added captions to {len(captioned_videos)} videos")
    if merge:
        logger.info(f"Final merged video created")
