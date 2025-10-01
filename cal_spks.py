# #!/usr/bin/env python3
# """
# Speaker Count Comparison Script
# Compares number of speakers between predicted RTTM and ground truth RTTM
# """

# import os
# import glob
# from collections import defaultdict

# def parse_rttm_speakers(rttm_path):
#     """
#     Parse RTTM file and return number of unique speakers per file_id.
#     RTTM format: SPEAKER <file-id> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
#     Returns: dict mapping file_id to set of speaker IDs
#     """
#     speakers = defaultdict(set)
    
#     with open(rttm_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith('#'):
#                 continue
            
#             parts = line.split()
#             if len(parts) < 8:
#                 continue
            
#             seg_type = parts[0]
#             if seg_type != "SPEAKER":
#                 continue
            
#             file_id = parts[1]
#             speaker_id = parts[7]
            
#             speakers[file_id].add(speaker_id)
    
#     return speakers

# def main():
#     import sys
    
#     # Allow command-line arguments or use defaults
#     if len(sys.argv) >= 3:
#         pred_dir = sys.argv[1]
#         ref_dir = sys.argv[2]
#     else:
#         # Try to auto-detect common locations
#         possible_pred_paths = ["scripts/after_clustering_v2"]
#         possible_ref_paths = ["dev"]
        
#         pred_dir = None
#         ref_dir = None
        
#         for path in possible_pred_paths:
#             if os.path.exists(path):
#                 pred_dir = path
#                 break
        
#         for path in possible_ref_paths:
#             if os.path.exists(path):
#                 ref_dir = path
#                 break
    
#     # Check if directories exist
#     if not pred_dir or not os.path.exists(pred_dir):
#         print(f"Error: Prediction directory not found!")
#         print(f"Usage: python {sys.argv[0]} <predicted_rttm_dir> <reference_dir>")
#         print(f"Example: python {sys.argv[0]} after_clustering dev")
#         return
    
#     if not ref_dir or not os.path.exists(ref_dir):
#         print(f"Error: Reference directory not found!")
#         print(f"Usage: python {sys.argv[0]} <predicted_rttm_dir> <reference_dir>")
#         print(f"Example: python {sys.argv[0]} after_clustering dev")
#         return
    
#     print(f"Using Predicted directory: {pred_dir}")
#     print(f"Using Reference directory: {ref_dir}")
#     print("=" * 100)
    
#     # Get all RTTM files
#     pred_files = glob.glob(os.path.join(pred_dir, "*.rttm"))
    
#     if not pred_files:
#         print(f"No RTTM files found in {pred_dir}")
#         return
    
#     print(f"\nFound {len(pred_files)} predicted RTTM files\n")
    
#     # Statistics
#     total_files = 0
#     correct_count = 0
#     over_estimated = 0
#     under_estimated = 0
    
#     errors = []
    
#     print(f"{'Filename':<40} {'Predicted':<12} {'Reference':<12} {'Difference':<12} {'Status':<15}")
#     print("-" * 100)
    
#     # Process each predicted file
#     for pred_path in sorted(pred_files[:]):
#         filename = os.path.basename(pred_path)
        
#         # Try different matching strategies
#         ref_path = os.path.join(ref_dir, filename)
#         print(f"./{ref_path}")
#         # If direct match doesn't exist, try removing common suffixes
#         #if not os.path.exists(ref_path):
#             # Try removing _diarization suffix
#             #base_name = filename.replace('_diarization.rttm', '.rttm')
#             #ref_path = os.path.join(ref_dir, base_name)
        
#         # Check if reference file exists
#         if not os.path.exists(ref_path):
#             print(f"{filename:<40} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'NO REFERENCE':<15}")
#             continue
        
#         # Parse both files
#         pred_speakers = parse_rttm_speakers(pred_path)
#         ref_speakers = parse_rttm_speakers(ref_path)
        
#         # Compare for each file_id in the RTTM
#         for file_id in pred_speakers.keys():
#             print("inside loop")
#             if file_id not in ref_speakers:
#                 continue
            
#             pred_count = len(pred_speakers[file_id])
#             ref_count = len(ref_speakers[file_id])
#             difference = pred_count - ref_count
            
#             total_files += 1
            
#             # Determine status
#             if difference == 0:
#                 status = "✓ CORRECT"
#                 correct_count += 1
#             elif difference > 0:
#                 status = "⚠ OVER"
#                 over_estimated += 1
#                 errors.append({
#                     'file': filename,
#                     'file_id': file_id,
#                     'pred': pred_count,
#                     'ref': ref_count,
#                     'diff': difference
#                 })
#             else:
#                 status = "⚠ UNDER"
#                 under_estimated += 1
#                 errors.append({
#                     'file': filename,
#                     'file_id': file_id,
#                     'pred': pred_count,
#                     'ref': ref_count,
#                     'diff': difference
#                 })
            
#             display_name = filename if len(filename) <= 40 else filename[:37] + "..."
#             print(f"{display_name:<40} {pred_count:<12} {ref_count:<12} {difference:+<12} {status:<15}")
#     print(total_files)
#     # Print summary
#     print("\n" + "=" * 100)
#     print("SUMMARY STATISTICS")
#     print("=" * 100)
    
#     if total_files > 0:
#         accuracy = (correct_count / total_files) * 100
#         over_rate = (over_estimated / total_files) * 100
#         under_rate = (under_estimated / total_files) * 100
        
#         print(f"\nTotal files compared: {total_files}")
#         print(f"Correct speaker count: {correct_count} ({accuracy:.2f}%)")
#         print(f"Over-estimated: {over_estimated} ({over_rate:.2f}%)")
#         print(f"Under-estimated: {under_estimated} ({under_rate:.2f}%)")
        
#         # Calculate average error
#         if errors:
#             avg_error = sum(abs(e['diff']) for e in errors) / len(errors)
#             max_over = max((e['diff'] for e in errors if e['diff'] > 0), default=0)
#             max_under = min((e['diff'] for e in errors if e['diff'] < 0), default=0)
            
#             print(f"\nAverage absolute error: {avg_error:.2f} speakers")
#             print(f"Maximum over-estimation: +{max_over} speakers")
#             print(f"Maximum under-estimation: {max_under} speakers")
        
#         # Show worst errors
#         if errors and len(errors) > 0:
#             print("\n" + "=" * 100)
#             print("TOP 10 WORST ERRORS")
#             print("=" * 100)
#             print(f"{'Filename':<40} {'Predicted':<12} {'Reference':<12} {'Difference':<12}")
#             print("-" * 100)
            
#             sorted_errors = sorted(errors, key=lambda x: abs(x['diff']), reverse=True)[:10]
#             for err in sorted_errors:
#                 display_name = err['file'] if len(err['file']) <= 40 else err['file'][:37] + "..."
#                 print(f"{display_name:<40} {err['pred']:<12} {err['ref']:<12} {err['diff']:+<12}")
#     else:
#         print("No valid files to compare!")
    
#     print("\n" + "=" * 100)

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
Speaker Count Comparison Script
Compares number of speakers between predicted RTTM and ground truth RTTM.
"""

import os
import glob
from collections import defaultdict

def parse_rttm_speakers(rttm_path):
    """
    Parse RTTM file and return set of unique speakers for the whole file.
    Normalizes file-id by stripping `.rttm` if present.
    """
    speakers_by_fileid = {}
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                file_id = parts[1]
                # normalize by removing .rttm if present
                file_id = file_id.replace(".rttm", "")
                speaker = parts[7]
                if file_id not in speakers_by_fileid:
                    speakers_by_fileid[file_id] = set()
                speakers_by_fileid[file_id].add(speaker)
    return speakers_by_fileid

def main():
    # Fixed paths
    pred_dir = "scripts/after_clustering_v2"
    ref_dir = "dev"
    
    if not os.path.exists(pred_dir):
        print(f"Error: Prediction directory not found: {pred_dir}")
        return
    if not os.path.exists(ref_dir):
        print(f"Error: Reference directory not found: {ref_dir}")
        return
    
    print(f"Using Predicted directory: {pred_dir}")
    print(f"Using Reference directory: {ref_dir}")
    print("=" * 100)
    
    pred_files = glob.glob(os.path.join(pred_dir, "*.rttm"))
    if not pred_files:
        print(f"No RTTM files found in {pred_dir}")
        return
    
    print(f"\nFound {len(pred_files)} predicted RTTM files\n")
    
    total_files = 0
    correct_count = 0
    over_estimated = 0
    under_estimated = 0
    errors = []
    
    print(f"{'Filename':<40} {'Predicted':<12} {'Reference':<12} {'Difference':<12} {'Status':<15}")
    print("-" * 100)
    
    for pred_path in sorted(pred_files):
        filename = os.path.basename(pred_path)
        ref_path = os.path.join(ref_dir, filename)

        if not os.path.exists(ref_path):
            print(f"{filename:<40} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'NO REFERENCE':<15}")
            continue
        
        pred_speakers = parse_rttm_speakers(pred_path)
        ref_speakers = parse_rttm_speakers(ref_path)
        
        for file_id in pred_speakers.keys():
            if file_id not in ref_speakers:
                continue
            
            pred_count = len(pred_speakers[file_id])
            ref_count = len(ref_speakers[file_id])
            difference = pred_count - ref_count
            
            total_files += 1
            if difference == 0:
                status = "✓ CORRECT"
                correct_count += 1
            elif difference > 0:
                status = "⚠ OVER"
                over_estimated += 1
                errors.append({'file': filename, 'file_id': file_id, 'pred': pred_count, 'ref': ref_count, 'diff': difference})
            else:
                status = "⚠ UNDER"
                under_estimated += 1
                errors.append({'file': filename, 'file_id': file_id, 'pred': pred_count, 'ref': ref_count, 'diff': difference})
            
            display_name = filename if len(filename) <= 40 else filename[:37] + "..."
            print(f"{display_name:<40} {pred_count:<12} {ref_count:<12} {difference:+<12} {status:<15}")
    
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    if total_files > 0:
        accuracy = (correct_count / total_files) * 100
        over_rate = (over_estimated / total_files) * 100
        under_rate = (under_estimated / total_files) * 100
        
        print(f"\nTotal files compared: {total_files}")
        print(f"Correct speaker count: {correct_count} ({accuracy:.2f}%)")
        print(f"Over-estimated: {over_estimated} ({over_rate:.2f}%)")
        print(f"Under-estimated: {under_estimated} ({under_rate:.2f}%)")
        
        if errors:
            avg_error = sum(abs(e['diff']) for e in errors) / len(errors)
            max_over = max((e['diff'] for e in errors if e['diff'] > 0), default=0)
            max_under = min((e['diff'] for e in errors if e['diff'] < 0), default=0)
            
            print(f"\nAverage absolute error: {avg_error:.2f} speakers")
            print(f"Maximum over-estimation: +{max_over} speakers")
            print(f"Maximum under-estimation: {max_under} speakers")
            
            print("\n" + "=" * 100)
            print("TOP 10 WORST ERRORS")
            print("=" * 100)
            print(f"{'Filename':<40} {'Predicted':<12} {'Reference':<12} {'Difference':<12}")
            print("-" * 100)
            sorted_errors = sorted(errors, key=lambda x: abs(x['diff']), reverse=True)[:10]
            for err in sorted_errors:
                display_name = err['file'] if len(err['file']) <= 40 else err['file'][:37] + "..."
                print(f"{display_name:<40} {err['pred']:<12} {err['ref']:<12} {err['diff']:+<12}")
    else:
        print("No valid files to compare!")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()
