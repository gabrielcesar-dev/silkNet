import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

import silknet.config as config

class SamplerSize(str, Enum):
    """Defines an enumerated type for the frame sampling options via the CLI."""
    small = "small"
    medium = "medium"
    large = "large"

SAMPLER_MAP = {
    SamplerSize.small: 1500,
    SamplerSize.medium: 3000,
    SamplerSize.large: 5000,
}

app = typer.Typer()


def setup_paths_and_validate(biome_name: str, sampler: SamplerSize) -> Tuple[Path, Path, int]:
    """
    Initializes and validates the directory paths and parameters for extraction.

    This function constructs the paths for the source (raw) and destination
    (processed) data based on the provided biome name. It validates the
    existence of the source directory and ensures the creation of the
    destination directory.

    Args:
        biome_name (str): The identifier for the biome to be processed.
        sampler (SamplerSize): The selected sampling option.

    Returns:
        Tuple[Path, Path, int]: A tuple containing the input directory path,
        the output directory path, and the total number of samples to be
        extracted.

    Raises:
        typer.Exit: Raised if the input directory is not found.
    """
    total_samples = SAMPLER_MAP[sampler]
    logger.info(f"Initiating process for biome: '{biome_name}'")
    logger.info(f"Sampler selected: '{sampler.value}' -> Total samples: {total_samples}")

    # Note: The original code used config.DATASET_NAME in the input path.
    # This might need adjustment based on the actual raw data structure.
    # Using a more common structure for raw data here.
    input_dir = config.RAW_DATA_DIR / config.DATASET_NAME / biome_name
    output_dir = config.PROCESSED_DATA_DIR / config.DATASET_NAME / biome_name

    logger.debug(f"Input directory (videos): {input_dir}")
    logger.debug(f"Output directory (images): {output_dir}")

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir, total_samples


def analyze_videos(input_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
    """
    Performs metadata analysis of video files within a directory.

    Iterates over the video files in the input directory to extract the
    path of each file and its respective total frame count. This information
    is crucial for the subsequent proportional allocation.

    Args:
        input_dir (Path): The absolute path to the source directory of the videos.

    Returns:
        Tuple[List[Dict[str, Any]], int]: A tuple containing a list of
        video metadata and the aggregated total frame count.

    Raises:
        typer.Exit: Raised if no video files are found.
    """
    logger.info("Analyzing videos to determine frame distribution...")
    video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
    video_files = [p for p in input_dir.iterdir() if p.suffix.lower() in video_extensions]

    if not video_files:
        logger.warning(f"No video files found in '{input_dir}'. Exiting.")
        raise typer.Exit()

    video_metadata = []
    grand_total_frames = 0
    for video_path in tqdm(video_files, desc="Analyzing videos"):
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                video_metadata.append({"path": video_path, "frames": frame_count})
                grand_total_frames += frame_count
            cap.release()
    
    logger.success(f"Analysis complete. Total of {grand_total_frames} frames available across {len(video_metadata)} videos.")
    return video_metadata, grand_total_frames


def allocate_and_extract_frames(video_metadata: list, grand_total_frames: int, total_samples: int, output_dir: Path) -> int:
    """
    Performs sample allocation and orchestrates the extraction process in parallel.

    This function utilizes a ProcessPoolExecutor to distribute the processing
    of each video to a different CPU core, achieving high parallelism for
    CPU-bound tasks.

    Args:
        video_metadata (list): The list of video metadata.
        grand_total_frames (int): The total sum of frames from all videos.
        total_samples (int): The total number of frames to be extracted.
        output_dir (Path): The destination directory for the images.

    Returns:
        int: The total number of frames that were effectively extracted.
    """
    logger.info("Allocating samples and preparing for parallel extraction...")
    tasks = []
    samples_allocated = 0

    for i, meta in enumerate(video_metadata):
        if i == len(video_metadata) - 1:
            num_to_extract = total_samples - samples_allocated
        else:
            proportion = meta["frames"] / grand_total_frames if grand_total_frames > 0 else 0
            num_to_extract = round(proportion * total_samples)
            samples_allocated += num_to_extract

        if num_to_extract > 0:
            tasks.append({
                "video_path": meta["path"],
                "output_dir": output_dir,
                "num_samples": num_to_extract
            })

    total_frames_extracted = 0
    num_videos = len(tasks)
    cpu_cores = os.cpu_count() or 4
    max_processes = min(num_videos, cpu_cores)
    
    logger.info(f"Found {num_videos} videos to process. Starting extraction with {max_processes} parallel processes...")

    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {executor.submit(process_single_video, **task): task for task in tasks}

        progress_bar = tqdm(as_completed(futures), total=len(tasks), desc="Processing videos")
        for future in progress_bar:
            task_info = futures[future]
            video_name = task_info['video_path'].name
            try:
                frames_saved = future.result()
                total_frames_extracted += frames_saved
                progress_bar.set_postfix_str(f"{video_name} -> {frames_saved} frames OK")
            except Exception as exc:
                logger.error(f"Task for video {video_name} generated an exception: {exc}")

    return total_frames_extracted


def process_single_video(video_path: Path, output_dir: Path, num_samples: int) -> int:
    """
    Executes frame extraction from a single video file.

    Implements the Reservoir Sampling algorithm to obtain a random selection
    of frames in a single sequential pass, optimizing performance by avoiding
    random seek operations. This version is memory-efficient, using temporary
    disk files as its reservoir instead of storing frames in RAM.

    Args:
        video_path (Path): The path to the video file to be processed.
        output_dir (Path): The final destination directory for the frames.
        num_samples (int): The number of frames to be extracted from this video.

    Returns:
        int: The number of frames that were successfully saved.
    """
    cap = cv2.VideoCapture(str(video_path), apiPreference=cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path.name}. Skipping.")
        return 0

    temp_dir = output_dir / f".{video_path.stem}_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    frames_saved_count = 0
    reservoir_paths = []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_samples:
            num_samples = total_frames
        
        if num_samples == 0:
            return 0
            
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)

            if len(reservoir_paths) < num_samples:
                temp_filepath = temp_dir / f"temp_frame_{frame_index:05d}.jpeg"
                cv2.imwrite(str(temp_filepath), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
                reservoir_paths.append(temp_filepath)
            else:
                j = random.randint(0, frame_index)
                if j < num_samples:
                    path_to_overwrite = reservoir_paths[j]
                    cv2.imwrite(str(path_to_overwrite), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

            frame_index += 1

        for i, temp_path in enumerate(reservoir_paths):
            final_filename = f"{video_path.stem}_{i:04d}.jpeg"
            final_filepath = output_dir / final_filename
            temp_path.rename(final_filepath)
            frames_saved_count += 1
            
    finally:
        cap.release()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    return frames_saved_count


@app.command()
def extract_frames_for_model(
    biome_name: str = typer.Argument(..., help="The name of the biome to be processed."),
    sampler: SamplerSize = typer.Argument(..., help="The sampler size (small, medium, large)."),
):
    """
    Orchestrates the complete pipeline for generating the image dataset.

    This CLI command function coordinates the sequential execution of the
    initialization, metadata analysis, and frame extraction steps,
    culminating in the creation of a processed dataset.
    """
    input_dir, output_dir, total_samples = setup_paths_and_validate(biome_name, sampler)

    video_metadata, grand_total_frames = analyze_videos(input_dir)
    
    if grand_total_frames < total_samples:
        logger.error(f"The total number of available frames ({grand_total_frames}) is less than the requested number of samples ({total_samples}).")
        raise typer.Exit(code=1)

    final_count = allocate_and_extract_frames(
        video_metadata=video_metadata,
        grand_total_frames=grand_total_frames,
        total_samples=total_samples,
        output_dir=output_dir
    )

    logger.success("Extraction process completed successfully!")
    logger.info(f"Total frames extracted: {final_count} (requested: {total_samples})")
    logger.info(f"All frames have been saved to: {output_dir}")


if __name__ == "__main__":
    app()