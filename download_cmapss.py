"""
Download CMAPSS (C-MAPSS) Turbofan Engine Degradation Dataset from NASA.

This script downloads the NASA Turbofan Engine Degradation Simulation Data Set,
which is used for predictive maintenance algorithm development and testing.

Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
"""
import requests
import zipfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config


# NASA Repository URL
CMAPSS_URL = "https://ti.arc.nasa.gov/c/6/"  # Main dataset download

# Alternative direct link (more reliable)
DIRECT_URL = "https://data.nasa.gov/download/xaut-bemq/application%2Fzip"

# Dataset information
DATASET_INFO = """
CMAPSS Dataset Information:
---------------------------
The C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset 
contains run-to-failure simulated data from turbofan engines.

Files included:
- train_FD001.txt: Training data for subset FD001
- test_FD001.txt: Test data for subset FD001
- RUL_FD001.txt: Remaining Useful Life labels for test set
- train_FD002.txt: Training data for subset FD002
- test_FD002.txt: Test data for subset FD002
- RUL_FD002.txt: RUL labels for FD002
- train_FD003.txt: Training data for subset FD003
- test_FD003.txt: Test data for subset FD003
- RUL_FD003.txt: RUL labels for FD003
- train_FD004.txt: Training data for subset FD004
- test_FD004.txt: Test data for subset FD004
- RUL_FD004.txt: RUL labels for FD004
- readme.txt: Dataset documentation

Data Format (space-separated):
- Column 1: Unit number
- Column 2: Time (in cycles)
- Columns 3-5: Operational settings
- Columns 6-26: Sensor measurements

Citation:
A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling 
for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the 
1st International Conference on Prognostics and Health Management (PHM08), 
Denver CO, Oct 2008.
"""


def download_cmapss(output_dir: Path = None, use_alternative: bool = True):
    """
    Download CMAPSS dataset.
    
    Args:
        output_dir: Directory to save data (defaults to Config.RAW_DATA_DIR)
        use_alternative: Use alternative download URL (more reliable)
    """
    if output_dir is None:
        output_dir = Config.RAW_DATA_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CMAPSS Dataset Downloader")
    print("=" * 60)
    print(DATASET_INFO)
    print("=" * 60)
    
    # Choose URL
    url = DIRECT_URL if use_alternative else CMAPSS_URL
    zip_path = output_dir / "CMAPSSData.zip"
    
    print(f"\n📥 Downloading from NASA repository...")
    print(f"   URL: {url}")
    print(f"   Destination: {zip_path}")
    
    try:
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n✓ Download complete: {zip_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: Manual download instructions:")
        print("1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print("2. Find 'Turbofan Engine Degradation Simulation Data Set'")
        print("3. Download 'CMAPSSData.zip'")
        print(f"4. Place in: {output_dir}")
        print("5. Run this script again to extract")
        
        if zip_path.exists():
            print("\n✓ Found existing zip file, attempting extraction...")
        else:
            return False
    
    # Extract
    if zip_path.exists():
        print(f"\n📦 Extracting files...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List contents
                file_list = zip_ref.namelist()
                print(f"   Found {len(file_list)} files in archive")
                
                # Extract all
                zip_ref.extractall(output_dir)
                
                # List extracted files
                print("\n✓ Extracted files:")
                for filename in file_list:
                    file_path = output_dir / filename
                    if file_path.exists():
                        size = file_path.stat().st_size
                        print(f"   - {filename} ({size:,} bytes)")
            
            print(f"\n✓ Extraction complete!")
            print(f"   Data directory: {output_dir}")
            
            # Verify key files
            key_files = ['train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt']
            missing = []
            
            for filename in key_files:
                if not (output_dir / filename).exists():
                    missing.append(filename)
            
            if missing:
                print(f"\n⚠️  Warning: Missing expected files: {missing}")
            else:
                print(f"\n✓ All key files present and ready to use!")
            
            # Optional: Remove zip file
            print(f"\n🗑️  Cleaning up...")
            zip_path.unlink()
            print(f"   Removed: {zip_path}")
            
            return True
            
        except zipfile.BadZipFile as e:
            print(f"\n✗ Extraction failed: {e}")
            print("The downloaded file may be corrupted. Try downloading again.")
            return False
    
    return False


def verify_data():
    """Verify the downloaded data is valid."""
    print("\n" + "=" * 60)
    print("Verifying Data")
    print("=" * 60)
    
    train_file = Config.RAW_DATA_DIR / 'train_FD001.txt'
    
    if not train_file.exists():
        print(f"✗ Training file not found: {train_file}")
        return False
    
    try:
        # Read first few lines
        with open(train_file, 'r') as f:
            lines = f.readlines()[:5]
        
        print(f"✓ File found: {train_file}")
        print(f"  Total size: {train_file.stat().st_size:,} bytes")
        print(f"\n  Sample data (first 5 lines):")
        for i, line in enumerate(lines, 1):
            # Show first 100 chars of each line
            print(f"  {i}: {line[:100].strip()}...")
        
        # Count lines
        with open(train_file, 'r') as f:
            line_count = sum(1 for _ in f)
        
        print(f"\n  Total records: {line_count:,}")
        
        # Parse a line to check format
        first_line = lines[0].strip().split()
        print(f"  Columns in first line: {len(first_line)}")
        
        if len(first_line) == 26:
            print("  ✓ Correct format (26 columns)")
        else:
            print(f"  ⚠️  Unexpected column count (expected 26, got {len(first_line)})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False


def main():
    """Main download pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download CMAPSS dataset from NASA')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/raw/)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data, do not download'
    )
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Show manual download instructions'
    )
    
    args = parser.parse_args()
    
    if args.manual:
        print(DATASET_INFO)
        print("\nManual Download Instructions:")
        print("1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print("2. Scroll to 'Turbofan Engine Degradation Simulation Data Set'")
        print("3. Click download link for 'CMAPSSData.zip'")
        print(f"4. Extract contents to: {Config.RAW_DATA_DIR}")
        print("5. Run: python download_cmapss.py --verify-only")
        return
    
    if args.verify_only:
        verify_data()
        return
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Config.RAW_DATA_DIR
    
    # Check if already exists
    train_file = output_dir / 'train_FD001.txt'
    if train_file.exists():
        print(f"✓ Dataset already exists at: {output_dir}")
        print(f"  Use --verify-only to check data integrity")
        print(f"  Delete files and re-run to re-download")
        verify_data()
        return
    
    # Download
    success = download_cmapss(output_dir)
    
    if success:
        # Verify
        verify_data()
        
        print("\n" + "=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Train models: python -m src.ml.trainer")
        print(f"  2. Build RAG index: python -m src.rag.index_builder")
        print(f"  3. Run verification: python verify_setup.py")
    else:
        print("\n" + "=" * 60)
        print("Download Failed")
        print("=" * 60)
        print("\nPlease try:")
        print("  - Manual download: python download_cmapss.py --manual")
        print("  - Check internet connection")
        print("  - Visit NASA repository directly")


if __name__ == "__main__":
    main()
