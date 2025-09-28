# fix_name.py
import os

def fix_rttm_filenames(rttm_dir):
    # Check if directory exists
    if not os.path.exists(rttm_dir):
        print(f"❌ Error: Directory '{rttm_dir}' does not exist")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Check if directory is empty
    files = os.listdir(rttm_dir)
    if not files:
        print(f"❌ Directory '{rttm_dir}' is empty")
        return
    
    # Find RTTM files
    rttm_files = [f for f in files if f.endswith("_diarization.rttm")]
    if not rttm_files:
        print(f"❌ No files ending with '_diarization.rttm' found in '{rttm_dir}'")
        print(f"Files found: {files}")
        return
    
    print(f"Found {len(rttm_files)} RTTM files to process")
    
    for file in rttm_files:
        filepath = os.path.join(rttm_dir, file)
        basename = file.replace("_diarization.rttm", "")

        fixed_lines = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1] == "<NA>":
                        parts[1] = basename
                    fixed_lines.append(" ".join(parts))

            with open(filepath, "w") as f:
                f.write("\n".join(fixed_lines))

            print(f"✅ Fixed {file} with basename '{basename}'")
        
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

# Try different possible paths
possible_paths = [
    "./scripts/after_clustering",
    "../scripts/after_clustering", 
    "./after_clustering",
    "/home/yagya/research_speakerdiarization/scripts/after_clustering"
]

print("Checking possible paths...")
for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Found directory: {path}")
        fix_rttm_filenames(path)
        break
else:
    print("❌ None of the expected paths exist:")
    for path in possible_paths:
        print(f"  - {path}")
    print(f"\nCurrent directory: {os.getcwd()}")
    print("Please check the correct path to your RTTM files.")