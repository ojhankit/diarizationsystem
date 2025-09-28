#!/usr/bin/env python3
import os
import csv
import argparse
from pyannote.core import Segment, Annotation, Timeline

def load_rttm(file_path, model=False):
    """Load RTTM file into pyannote Annotation object."""
    ann = Annotation()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            file_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            spk_label = parts[7]
            if model:
                # convert model labels SPEAKER_00 -> spk00
                spk_label = "spk" + spk_label.split("_")[1]
            ann[Segment(start, start + duration)] = spk_label
    return ann

def map_speaker_labels(truth_ann, model_ann):
    """
    Map model speakers to truth speakers based on maximum overlap.
    Returns new Annotation with mapped labels.
    """
    mapping = {}
    used_model_spks = set()
    
    for t_spk in truth_ann.labels():
        best_overlap = 0
        best_model_spk = None
        t_timeline = truth_ann.label_timeline(t_spk)
        
        for m_spk in model_ann.labels():
            if m_spk in used_model_spks:
                continue
            m_timeline = model_ann.label_timeline(m_spk)
            
            # Compute overlap duration manually
            overlap_duration = 0.0
            for t_seg in t_timeline:
                for m_seg in m_timeline:
                    start = max(t_seg.start, m_seg.start)
                    end = min(t_seg.end, m_seg.end)
                    if end > start:
                        overlap_duration += end - start
            
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_model_spk = m_spk
                
        if best_model_spk:
            mapping[best_model_spk] = t_spk
            used_model_spks.add(best_model_spk)

    # Remap model annotation
    new_model_ann = Annotation()
    for segment, _, spk in model_ann.itertracks(yield_label=True):
        mapped_spk = mapping.get(spk, spk)
        new_model_ann[segment] = mapped_spk
    return new_model_ann

# def compute_der(truth_ann, model_ann):
    """
    Compute DER: (Miss + False Alarm + Confusion) / total_time
    Uses proper DER calculation accounting for missed speech, false alarms, and confusion.
    """
    # Get total reference time
    total_time = 0.0
    for segment, _, _ in truth_ann.itertracks(yield_label=True):
        total_time += segment.duration
    
    if total_time == 0:
        # If no reference speech, DER is 0 if no hypothesis speech, 1 otherwise
        return 1.0 if len(model_ann) > 0 else 0.0

    error_time = 0.0

    # Calculate missed speech and confusion
    for segment, _, t_spk in truth_ann.itertracks(yield_label=True):
        segment_duration = segment.duration
        
        # Find overlapping model segments
        overlapping = model_ann.crop(segment)
        
        if len(overlapping) == 0:
            # Complete miss
            error_time += segment_duration
        else:
            # Check overlapping parts for confusion and partial misses
            covered_duration = 0.0
            confusion_duration = 0.0
            
            for o_segment, _, m_spk in overlapping.itertracks(yield_label=True):
                # Calculate intersection duration
                start = max(segment.start, o_segment.start)
                end = min(segment.end, o_segment.end)
                if end > start:
                    intersection_duration = end - start
                    covered_duration += intersection_duration
                    
                    if t_spk != m_spk:
                        # Speaker confusion
                        confusion_duration += intersection_duration
            
            # Add confusion time to errors
            error_time += confusion_duration
            
            # Add uncovered time as missed speech
            uncovered_duration = segment_duration - covered_duration
            if uncovered_duration > 0:
                error_time += uncovered_duration

    # Calculate false alarms
    for segment, _, _ in model_ann.itertracks(yield_label=True):
        segment_duration = segment.duration
        
        # Find overlapping truth segments
        overlapping_truth = truth_ann.crop(segment)
        
        if len(overlapping_truth) == 0:
            # Complete false alarm
            error_time += segment_duration
        else:
            # Check for partial false alarms
            covered_duration = 0.0
            
            for t_segment, _, _ in overlapping_truth.itertracks(yield_label=True):
                # Calculate intersection duration
                start = max(segment.start, t_segment.start)
                end = min(segment.end, t_segment.end)
                if end > start:
                    covered_duration += end - start
            
            # Add uncovered time as false alarm
            uncovered_duration = segment_duration - covered_duration
            if uncovered_duration > 0:
                error_time += uncovered_duration

    return error_time / total_time
def compute_der_components(truth_ann, model_ann):
    """
    Compute DER components: Missed speech, False Alarms, Confusion
    Returns (missed, false_alarm, confusion, total_time)
    """
    total_time = sum(seg.duration for seg, _, _ in truth_ann.itertracks(yield_label=True))
    if total_time == 0:
        # If no reference speech, DER is 0 if no hypothesis speech, 1 otherwise
        fa = sum(seg.duration for seg, _, _ in model_ann.itertracks(yield_label=True))
        return 0.0, fa, 0.0, 0.0

    missed = 0.0
    confusion = 0.0
    false_alarm = 0.0

    # Missed speech and confusion
    for t_seg, _, t_spk in truth_ann.itertracks(yield_label=True):
        overlapping = model_ann.crop(t_seg)
        if len(overlapping) == 0:
            missed += t_seg.duration
        else:
            covered = 0.0
            for o_seg, _, m_spk in overlapping.itertracks(yield_label=True):
                start = max(t_seg.start, o_seg.start)
                end = min(t_seg.end, o_seg.end)
                if end > start:
                    intersection = end - start
                    covered += intersection
                    if t_spk != m_spk:
                        confusion += intersection
            missed += t_seg.duration - covered

    # False Alarms
    for m_seg, _, _ in model_ann.itertracks(yield_label=True):
        overlapping_truth = truth_ann.crop(m_seg)
        if len(overlapping_truth) == 0:
            false_alarm += m_seg.duration
        else:
            covered = 0.0
            for t_seg, _, _ in overlapping_truth.itertracks(yield_label=True):
                start = max(m_seg.start, t_seg.start)
                end = min(m_seg.end, t_seg.end)
                if end > start:
                    covered += end - start
            false_alarm += m_seg.duration - covered

    return missed, false_alarm, confusion, total_time

def compute_jer(truth_ann, model_ann):
    """
    Compute JER (Jaccard Error Rate) for each speaker and average.
    JER = 1 - (intersection / union) for each speaker.
    """
    jer_list = []
    
    # Get all unique speakers from both annotations
    all_speakers = set(truth_ann.labels()) | set(model_ann.labels())
    
    for spk in all_speakers:
        # Get timelines for this speaker
        t_timeline = truth_ann.label_timeline(spk) if spk in truth_ann.labels() else Timeline()
        m_timeline = model_ann.label_timeline(spk) if spk in model_ann.labels() else Timeline()
        
        # Compute intersection duration manually
        intersection_duration = 0.0
        for t_seg in t_timeline:
            for m_seg in m_timeline:
                start = max(t_seg.start, m_seg.start)
                end = min(t_seg.end, m_seg.end)
                if end > start:
                    intersection_duration += end - start
        
        # Compute union duration manually
        union_duration = t_timeline.duration() + m_timeline.duration() - intersection_duration
        
        if union_duration == 0:
            jer = 0.0  # Both timelines are empty
        else:
            jer = 1.0 - (intersection_duration / union_duration)
        
        jer_list.append(jer)
    
    return sum(jer_list) / len(jer_list) if jer_list else 0.0

def main(truth_dir, model_dir, output_csv):
    results = []
    print(f"{'File':20s} {'DER':>6s} {'MISS':>6s} {'FA':>6s} {'CONF':>6s} {'JER':>6s}")
    for file_name in os.listdir(truth_dir):
        if not file_name.endswith(".rttm"):
            continue
        file_id = file_name.replace(".rttm", "")
        truth_path = os.path.join(truth_dir, file_name)
        model_path = os.path.join(model_dir, f"{file_id}_diarization.rttm")
        if not os.path.exists(model_path):
            continue

        truth_ann = load_rttm(truth_path)
        model_ann = load_rttm(model_path, model=True)
        model_ann_mapped = map_speaker_labels(truth_ann, model_ann)

        # DER components
        miss, fa, conf, total_time = compute_der_components(truth_ann, model_ann_mapped)
        der = (miss + fa + conf) / total_time if total_time > 0 else 0.0
        jer = compute_jer(truth_ann, model_ann_mapped)

        results.append([file_id, der, miss, fa, conf, jer])
        print(f"{file_id:20s} {der:6.3f} {miss:6.3f} {fa:6.3f} {conf:6.3f} {jer:6.3f}")

# Print averages
    if results:
        avg_der = sum(r[1] for r in results) / len(results)
        avg_miss = sum(r[2] for r in results) / len(results)
        avg_fa = sum(r[3] for r in results) / len(results)
        avg_conf = sum(r[4] for r in results) / len(results)
        avg_jer = sum(r[5] for r in results) / len(results)
        print("\nAverage over all files:")
        #print(f"{'Average':20s} {avg_der:6.3f} {avg_miss:6.3f} {avg_fa:6.3f} {avg_conf:6.3f} {avg_jer:6.3f}")

        print(f"Average False Alarm: {avg_fa:6.3f}")
        print(f"Average Missed Speech: {avg_miss:6.3f}")
        print(f"Average Confusiin: {avg_conf:6.3f}")
        print(f"Average DER: {avg_der:6.3f}")
        print(f"Average JER: {avg_jer:6.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate speaker diarization using DER and JER metrics")
    parser.add_argument("--truth_dir", required=True, help="Directory containing ground truth RTTM files")
    parser.add_argument("--model_dir", required=True, help="Directory containing model prediction RTTM files")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    main(args.truth_dir, args.model_dir, args.output_csv)