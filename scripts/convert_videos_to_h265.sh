#!/bin/bash
#
# Robust Batch Video Standardizer for AI Datasets
#
# Description:
#   This script standardizes all videos in a source directory to a format
#   suitable for AI training (720p, 30 FPS, H.265, no audio).
#   It is designed to skip any corrupted files and report a summary at the end.
#
# Usage:
#   Run this script from the root of the project.
#   ./scripts/convert_videos.sh
#

set -uo pipefail

# --- Configuration ---
readonly INPUT_DIR="data/raw/hollow_knight_silksong_dataset"
readonly OUTPUT_DIR="data/interim/hollow_knight_silksong_dataset_720p_30fps"

# FFmpeg Options for Standardization:
# -vf "scale=-1:720":  Scale video to 720p height, keeping aspect ratio.
# -r 30:               Set framerate to 30 FPS.
# -c:v libx265:        Use the efficient H.265 (HEVC) video codec.
# -crf 28:             Quality factor for H.265 (good balance).
# -preset medium:      A good balance between conversion speed and file size.
# -an:                 No Audio. Removes the audio track.
readonly FFMPEG_OPTS='-vf scale=-1:720 -r 30 -c:v libx265 -crf 28 -preset medium -an'

# --- Pre-flight Checks ---
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: FFmpeg is not installed. Please install it to continue." >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Source directory not found at '$INPUT_DIR'." >&2
    exit 1
fi

# --- Main Logic ---
echo "Starting robust video standardization..."
echo "Source:      ${INPUT_DIR}"
echo "Destination: ${OUTPUT_DIR}"

mapfile -t video_files < <(find "$INPUT_DIR" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" \))

if [[ ${#video_files[@]} -eq 0 ]]; then
    echo "Warning: No video files found in the source directory."
    exit 0
fi

total_files=${#video_files[@]}
success_count=0
fail_count=0
echo "Found ${total_files} videos to process."
mkdir -p "$OUTPUT_DIR"

# Process each video file
for i in "${!video_files[@]}"; do
    input_file="${video_files[$i]}"
    relative_path="${input_file#$INPUT_DIR/}"
    output_file="$OUTPUT_DIR/$relative_path"
    
    mkdir -p "$(dirname "$output_file")"

    echo -ne "[$((i + 1))/${total_files}] Standardizing ${input_file}..."

    # Execute FFmpeg. The 'if' statement directly checks the command's success.
    # We save FFmpeg's error output to a variable to show it if something fails.
    if error_output=$(ffmpeg -y -i "$input_file" ${FFMPEG_OPTS} "$output_file" -hide_banner -loglevel error 2>&1); then
        echo " DONE"
        ((success_count++))
    else
        echo " FAILED"
        echo "    └─ Reason: ${error_output}"
        ((fail_count++))
    fi
done

echo "--------------------------------------------------"
echo "Standardization process completed."
echo "Summary:"
echo "Successful: ${success_count}"
echo "Failed:     ${fail_count}"