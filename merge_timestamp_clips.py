import json
import argparse
import os

def merge_overlapping_clips(clips: list[dict]) -> list[dict]:
    if not clips or len(clips) <= 1:
        return clips

    # Sort clips by start_time
    sorted_clips = sorted(clips, key=lambda x: x['start_time'])

    merged_clips = []
    if not sorted_clips: # Should not happen if initial check passed, but defensive
        return []

    current_merged_clip = dict(sorted_clips[0]) # Start with a copy of the first clip

    for i in range(1, len(sorted_clips)):
        next_clip = sorted_clips[i]
        # Check for overlap or adjacency (allowing a small gap, e.g., 0.1s, could be an option but sticking to direct overlap for now)
        if next_clip['start_time'] <= current_merged_clip['end_time']: 
            # Merge
            current_merged_clip['end_time'] = max(current_merged_clip['end_time'], next_clip['end_time'])
            # Concatenate transcript snippets
            current_merged_clip['transcript_snippet'] += " ... " + next_clip.get('transcript_snippet', '')
        else:
            # No overlap, finalize the current_merged_clip and start a new one
            merged_clips.append(current_merged_clip)
            current_merged_clip = dict(next_clip) # Start new merge with a copy

    merged_clips.append(current_merged_clip) # Add the last processed clip

    return merged_clips

def main():
    parser = argparse.ArgumentParser(description="Merges overlapping video clips in a timestamp JSON file.")
    parser.add_argument("input_file", help="Path to the input _timestamps.json file.")
    args = parser.parse_args()

    input_filepath = args.input_file

    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found: {input_filepath}")
        return

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            timestamp_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_filepath}: {e}")
        return
    except Exception as e:
        print(f"Error reading file {input_filepath}: {e}")
        return

    if not isinstance(timestamp_data, list):
        print(f"Error: Expected a list of timestamp entries in {input_filepath}")
        return

    processed_timestamp_data = []
    for entry in timestamp_data:
        if not isinstance(entry, dict):
            processed_timestamp_data.append(entry) # Keep non-dict entries as is
            continue
        
        original_clips = entry.get("video_clips")
        if isinstance(original_clips, list):
            merged_video_clips = merge_overlapping_clips(original_clips)
            new_entry = dict(entry) # Make a copy to modify
            new_entry["video_clips"] = merged_video_clips
            processed_timestamp_data.append(new_entry)
        else:
            processed_timestamp_data.append(entry) # Keep entry as is if "video_clips" is not a list

    # Determine output filepath
    base, ext = os.path.splitext(input_filepath)
    output_filepath = f"{base}_merged{ext}"

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_timestamp_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully processed timestamps. Merged clips saved to: {output_filepath}")
    except Exception as e:
        print(f"Error writing output file {output_filepath}: {e}")

if __name__ == "__main__":
    main() 