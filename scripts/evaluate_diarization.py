
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
            start = float(parts[3])
            duration = float(parts[4])
            spk_label = parts[7]
            #if model:
            #    spk_label = "spk" + spk_label.split("_")[1]
            ann[Segment(start, start + duration)] = spk_label
    return ann

def map_speaker_labels(truth_ann, model_ann):
    """Map model speakers to truth speakers using maximum overlap."""
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
            overlap_duration = 0.0
            for t_seg in t_timeline:
                for m_seg in m_timeline:
                    start = max(t_seg.start, m_seg.start)
                    end = min(t_seg.end, m_seg.end)
                    overlap_duration += max(0.0, end - start)
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_model_spk = m_spk
        if best_model_spk:
            mapping[best_model_spk] = t_spk
            used_model_spks.add(best_model_spk)

    new_model_ann = Annotation()
    for segment, _, spk in model_ann.itertracks(yield_label=True):
        mapped_spk = mapping.get(spk, spk)
        new_model_ann[segment] = mapped_spk
    return new_model_ann

def compute_der_components(truth_ann, model_ann):
    """
    Compute DER components: Missed speech, False Alarms, Confusion
    Returns (missed, false_alarm, confusion, total_time)
    """
    total_time = sum(seg.duration for seg, _, _ in truth_ann.itertracks(yield_label=True))
    if total_time == 0:
        # No reference speech
        fa = sum(seg.duration for seg, _, _ in model_ann.itertracks(yield_label=True))
        return 0.0, fa, 0.0, 0.0

    missed = 0.0
    confusion = 0.0
    false_alarm = 0.0

    # Missed speech & confusion
    for t_seg, _, t_spk in truth_ann.itertracks(yield_label=True):
        overlapping = model_ann.crop(t_seg)
        if len(overlapping) == 0:
            missed += t_seg.duration
        else:
            covered = 0.0
            for o_seg, _, m_spk in overlapping.itertracks(yield_label=True):
                start = max(t_seg.start, o_seg.start)
                end = min(t_seg.end, o_seg.end)
                intersection = max(0.0, end - start)
                covered += intersection
                if t_spk != m_spk:
                    confusion += intersection
            missed += max(0.0, t_seg.duration - covered)

    # False alarms
    for m_seg, _, _ in model_ann.itertracks(yield_label=True):
        overlapping_truth = truth_ann.crop(m_seg)
        if len(overlapping_truth) == 0:
            false_alarm += m_seg.duration
        else:
            covered = 0.0
            for t_seg, _, _ in overlapping_truth.itertracks(yield_label=True):
                start = max(m_seg.start, t_seg.start)
                end = min(m_seg.end, t_seg.end)
                covered += max(0.0, end - start)
            false_alarm += max(0.0, m_seg.duration - covered)

    return missed, false_alarm, confusion, total_time

def compute_jer(truth_ann, model_ann):
    """Compute average Jaccard Error Rate (JER) over all speakers."""
    jer_list = []
    all_speakers = set(truth_ann.labels()) | set(model_ann.labels())
    for spk in all_speakers:
        t_timeline = truth_ann.label_timeline(spk) if spk in truth_ann.labels() else Timeline()
        m_timeline = model_ann.label_timeline(spk) if spk in model_ann.labels() else Timeline()

        # Intersection
        intersection = 0.0
        for t_seg in t_timeline:
            for m_seg in m_timeline:
                start = max(t_seg.start, m_seg.start)
                end = min(t_seg.end, m_seg.end)
                intersection += max(0.0, end - start)

        # Union
        union = t_timeline.duration() + m_timeline.duration() - intersection
        jer = 0.0 if union == 0 else 1.0 - (intersection / union)
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
        model_path = os.path.join(model_dir, f"{file_id}.rttm")
        if not os.path.exists(model_path):
            continue

        truth_ann = load_rttm(truth_path)
        model_ann = load_rttm(model_path, model=True)
        model_ann_mapped = map_speaker_labels(truth_ann, model_ann)

        miss, fa, conf, total_time = compute_der_components(truth_ann, model_ann_mapped)
        der = (miss + fa + conf) / total_time if total_time > 0 else 0.0
        jer = compute_jer(truth_ann, model_ann_mapped)

        # Clip negative values just in case
        miss = max(0.0, miss)
        fa = max(0.0, fa)
        conf = max(0.0, conf)

        results.append([file_id, der, miss, fa, conf, jer])
        print(f"{file_id:20s} {der:6.3f} {miss:6.3f} {fa:6.3f} {conf:6.3f} {jer:6.3f}")

    # Averages
    if results:
        avg_der = sum(r[1] for r in results) / len(results)
        avg_miss = sum(r[2] for r in results) / len(results)
        avg_fa = sum(r[3] for r in results) / len(results)
        avg_conf = sum(r[4] for r in results) / len(results)
        avg_jer = sum(r[5] for r in results) / len(results)
        print("\nAverage over all files:")
        print(f"Average False Alarm: {avg_fa:6.3f}")
        print(f"Average Missed Speech: {avg_miss:6.3f}")
        print(f"Average Confusion: {avg_conf:6.3f}")
        print(f"Average DER: {avg_der:6.3f}")
        print(f"Average JER: {avg_jer:6.3f}")

        # Write CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "der", "miss", "fa", "conf", "jer"])
            writer.writerows(results)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate speaker diarization using DER and JER metrics")
    parser.add_argument("--truth_dir", required=True, help="Directory containing ground truth RTTM files")
    parser.add_argument("--model_dir", required=True, help="Directory containing model prediction RTTM files")
    parser.add_argument("--output_csv", required=True, help="Output CSV file path")
    args = parser.parse_args()
    main(args.truth_dir, args.model_dir, args.output_csv)
 