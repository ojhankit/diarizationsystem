#!/usr/bin/env python3
"""
VAD Error Rate Calculator
Compares VAD output (RTTM) against ground truth diarization (RTTM)
Calculates False Alarm, Missed Speech, and VAD Error Rate
"""

import os
import glob
from collections import defaultdict
import numpy as np

def parse_rttm(rttm_path):
    """
    Parse RTTM file and return speech segments.
    RTTM format: SPEAKER <file-id> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    Returns: dict mapping file_id to list of (start, end) tuples
    """
    segments = defaultdict(list)
    
    with open(rttm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            seg_type = parts[0]
            if seg_type != "SPEAKER":
                continue
            
            file_id = parts[1]
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            
            segments[file_id].append((start_time, end_time))
    
    return segments

def merge_overlapping_segments(segments):
    """
    Merge overlapping segments to get total speech regions.
    For VAD, we don't care about speaker identity, just speech/non-speech.
    """
    if not segments:
        return []
    
    # Sort by start time
    sorted_segs = sorted(segments)
    merged = [sorted_segs[0]]
    
    for current_start, current_end in sorted_segs[1:]:
        last_start, last_end = merged[-1]
        
        # If overlapping or adjacent, merge
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    
    return merged

def get_total_duration(segments):
    """Calculate total duration of speech segments."""
    return sum(end - start for start, end in segments)

def calculate_overlap(seg1_list, seg2_list):
    """
    Calculate overlap duration between two lists of segments.
    """
    overlap = 0.0
    
    for start1, end1 in seg1_list:
        for start2, end2 in seg2_list:
            # Calculate intersection
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap += (overlap_end - overlap_start)
    
    return overlap

def calculate_vad_metrics(vad_segments, ref_segments):
    """
    Calculate VAD error metrics.
    
    False Alarm (FA): VAD detects speech when there is none
    Missed Speech (MS): VAD misses speech that exists
    VAD Error Rate = (FA + MS) / Total Reference Speech Duration
    
    Returns: dict with metrics
    """
    # Merge overlapping segments for both VAD and reference
    vad_merged = merge_overlapping_segments(vad_segments)
    ref_merged = merge_overlapping_segments(ref_segments)
    
    # Calculate durations
    vad_duration = get_total_duration(vad_merged)
    ref_duration = get_total_duration(ref_merged)
    
    # Calculate correct detection (overlap between VAD and reference)
    correct_duration = calculate_overlap(vad_merged, ref_merged)
    
    # Calculate errors
    false_alarm = vad_duration - correct_duration  # VAD detected but no reference
    missed_speech = ref_duration - correct_duration  # Reference exists but VAD missed
    
    # Calculate rates
    if ref_duration > 0:
        false_alarm_rate = (false_alarm / ref_duration) * 100
        missed_speech_rate = (missed_speech / ref_duration) * 100
        vad_error_rate = ((false_alarm + missed_speech) / ref_duration) * 100
    else:
        false_alarm_rate = 0.0
        missed_speech_rate = 0.0
        vad_error_rate = 0.0
    
    return {
        'vad_duration': vad_duration,
        'ref_duration': ref_duration,
        'correct_duration': correct_duration,
        'false_alarm': false_alarm,
        'missed_speech': missed_speech,
        'false_alarm_rate': false_alarm_rate,
        'missed_speech_rate': missed_speech_rate,
        'vad_error_rate': vad_error_rate
    }

def main():
    vad_dir = "scripts/vad_outputs"
    ref_dir = "dev"
    
    # Check if directories exist
    if not os.path.exists(vad_dir):
        print(f"Error: VAD directory '{vad_dir}' not found!")
        return
    
    if not os.path.exists(ref_dir):
        print(f"Error: Reference directory '{ref_dir}' not found!")
        return
    
    # Get all RTTM files
    vad_files = glob.glob(os.path.join(vad_dir, "*.rttm"))
    
    if not vad_files:
        print(f"No RTTM files found in {vad_dir}")
        return
    
    print(f"Found {len(vad_files)} VAD output files")
    print("=" * 80)
    
    # Aggregate metrics
    total_metrics = {
        'vad_duration': 0.0,
        'ref_duration': 0.0,
        'correct_duration': 0.0,
        'false_alarm': 0.0,
        'missed_speech': 0.0
    }
    
    file_results = []
    
    # Process each VAD file
    for vad_path in sorted(vad_files):
        vad_filename = os.path.basename(vad_path)
        ref_path = os.path.join(ref_dir, vad_filename)
        
        # Check if reference file exists
        if not os.path.exists(ref_path):
            print(f"Warning: Reference file not found for {vad_filename}, skipping...")
            continue
        
        # Parse both files
        vad_data = parse_rttm(vad_path)
        ref_data = parse_rttm(ref_path)
        
        # Process each file_id in the RTTM
        for file_id in vad_data.keys():
            if file_id not in ref_data:
                print(f"Warning: File ID '{file_id}' not found in reference, skipping...")
                continue
            
            vad_segments = vad_data[file_id]
            ref_segments = ref_data[file_id]
            
            # Calculate metrics
            metrics = calculate_vad_metrics(vad_segments, ref_segments)
            
            # Store results
            file_results.append({
                'file': vad_filename,
                'file_id': file_id,
                'metrics': metrics
            })
            
            # Aggregate
            for key in total_metrics.keys():
                total_metrics[key] += metrics[key]
            
            # Print per-file results
            print(f"\nFile: {vad_filename} (ID: {file_id})")
            print(f"  Reference Speech Duration: {metrics['ref_duration']:.2f}s")
            print(f"  VAD Speech Duration: {metrics['vad_duration']:.2f}s")
            print(f"  Correct Detection: {metrics['correct_duration']:.2f}s")
            print(f"  False Alarm: {metrics['false_alarm']:.2f}s ({metrics['false_alarm_rate']:.2f}%)")
            print(f"  Missed Speech: {metrics['missed_speech']:.2f}s ({metrics['missed_speech_rate']:.2f}%)")
            print(f"  VAD Error Rate: {metrics['vad_error_rate']:.2f}%")
    
    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("OVERALL VAD PERFORMANCE")
    print("=" * 80)
    
    if total_metrics['ref_duration'] > 0:
        overall_fa_rate = (total_metrics['false_alarm'] / total_metrics['ref_duration']) * 100
        overall_ms_rate = (total_metrics['missed_speech'] / total_metrics['ref_duration']) * 100
        overall_ver = ((total_metrics['false_alarm'] + total_metrics['missed_speech']) / 
                       total_metrics['ref_duration']) * 100
        
        print(f"\nTotal Reference Speech Duration: {total_metrics['ref_duration']:.2f}s")
        print(f"Total VAD Speech Duration: {total_metrics['vad_duration']:.2f}s")
        print(f"Total Correct Detection: {total_metrics['correct_duration']:.2f}s")
        print(f"\nTotal False Alarm: {total_metrics['false_alarm']:.2f}s ({overall_fa_rate:.2f}%)")
        print(f"Total Missed Speech: {total_metrics['missed_speech']:.2f}s ({overall_ms_rate:.2f}%)")
        print(f"\n*** Overall VAD Error Rate (VER): {overall_ver:.2f}% ***")
        
        # Additional statistics
        print(f"\nNumber of files processed: {len(file_results)}")
        print(f"Average file duration: {total_metrics['ref_duration']/len(file_results):.2f}s")
    else:
        print("No valid data to calculate metrics!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()