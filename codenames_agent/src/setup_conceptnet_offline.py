"""
Setup Script for ConceptNet Offline Data
=========================================
Downloads and processes ConceptNet data for offline use.

This script will:
1. Download the ConceptNet assertions file (~500MB compressed)
2. Extract English-only edges
3. Save as pickle for fast loading (~100MB)

Total time: 10-15 minutes (depending on internet speed)
Disk space needed: ~2GB during processing, ~100MB final

Usage:
    python src/setup_conceptnet_offline.py
"""

import os
import sys
import gzip
import pickle
import urllib.request
import shutil
from collections import defaultdict
import json

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CONCEPTNET_CSV_GZ = os.path.join(DATA_DIR, "conceptnet-assertions-5.7.0.csv.gz")
CONCEPTNET_ENGLISH_PKL = os.path.join(DATA_DIR, "conceptnet_english.pkl")


def download_progress(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  Downloading: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)")
    sys.stdout.flush()


def download_conceptnet():
    """Download ConceptNet assertions file."""
    print("\n" + "=" * 60)
    print("STEP 1: Downloading ConceptNet Data")
    print("=" * 60)
    print(f"URL: {CONCEPTNET_URL}")
    print(f"Destination: {CONCEPTNET_CSV_GZ}")
    print("File size: ~500 MB (this may take a few minutes)")

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CONCEPTNET_CSV_GZ):
        size = os.path.getsize(CONCEPTNET_CSV_GZ)
        if size > 400_000_000:  # ~400MB minimum
            print(f"\nFile already exists ({size / 1024 / 1024:.1f} MB). Skipping download.")
            return True
        else:
            print(f"\nIncomplete file found ({size / 1024 / 1024:.1f} MB). Re-downloading...")
            os.remove(CONCEPTNET_CSV_GZ)

    try:
        print("\nStarting download...")
        urllib.request.urlretrieve(CONCEPTNET_URL, CONCEPTNET_CSV_GZ, download_progress)
        print("\n  Download complete!")
        return True
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  {CONCEPTNET_URL}")
        print(f"  Save to: {CONCEPTNET_CSV_GZ}")
        return False


def process_conceptnet():
    """Process ConceptNet data and extract English edges."""
    print("\n" + "=" * 60)
    print("STEP 2: Processing ConceptNet Data")
    print("=" * 60)
    print("Extracting English-only edges (this may take 5-10 minutes)...")

    if os.path.exists(CONCEPTNET_ENGLISH_PKL):
        size = os.path.getsize(CONCEPTNET_ENGLISH_PKL)
        print(f"\nProcessed file already exists ({size / 1024 / 1024:.1f} MB).")
        response = input("Reprocess? (y/N): ").strip().lower()
        if response != 'y':
            print("Using existing file.")
            return True

    edges = defaultdict(list)
    relations = defaultdict(list)
    words = set()

    line_count = 0
    english_count = 0

    try:
        with gzip.open(CONCEPTNET_CSV_GZ, 'rt', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 1000000 == 0:
                    print(f"  Processed {line_count:,} lines, found {english_count:,} English edges...")

                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                # ConceptNet CSV format:
                # URI, relation, start, end, metadata_json
                relation_uri = parts[1]
                start_uri = parts[2]
                end_uri = parts[3]

                # Filter for English only
                if '/c/en/' not in start_uri or '/c/en/' not in end_uri:
                    continue

                # Extract words and relation
                try:
                    start_word = start_uri.split('/')[3].replace('_', ' ').lower()
                    end_word = end_uri.split('/')[3].replace('_', ' ').lower()
                    relation = relation_uri.split('/')[-1]

                    # Skip multi-word phrases (longer than 3 words)
                    if start_word.count(' ') > 2 or end_word.count(' ') > 2:
                        continue

                    # Get weight
                    weight = 1.0
                    try:
                        metadata = json.loads(parts[4])
                        weight = metadata.get('weight', 1.0)
                    except:
                        pass

                    # Store edges
                    edges[start_word].append((relation, end_word, weight))
                    edges[end_word].append((relation, start_word, weight))
                    relations[relation].append((start_word, end_word, weight))
                    words.add(start_word)
                    words.add(end_word)
                    english_count += 1

                except (IndexError, KeyError):
                    continue

        print(f"\n  Total lines processed: {line_count:,}")
        print(f"  English edges found: {english_count:,}")
        print(f"  Unique words: {len(words):,}")
        print(f"  Unique relations: {len(relations)}")

        # Save as pickle
        print(f"\n  Saving to: {CONCEPTNET_ENGLISH_PKL}")
        data = {
            'edges': dict(edges),
            'relations': dict(relations),
            'words': words
        }
        with open(CONCEPTNET_ENGLISH_PKL, 'wb') as f:
            pickle.dump(data, f)

        size = os.path.getsize(CONCEPTNET_ENGLISH_PKL)
        print(f"  Saved! File size: {size / 1024 / 1024:.1f} MB")

        return True

    except Exception as e:
        print(f"\n  Error processing: {e}")
        return False


def verify_setup():
    """Verify the setup is complete and working."""
    print("\n" + "=" * 60)
    print("STEP 3: Verifying Setup")
    print("=" * 60)

    if not os.path.exists(CONCEPTNET_ENGLISH_PKL):
        print("  ERROR: Processed file not found!")
        return False

    print("  Loading processed data...")
    try:
        with open(CONCEPTNET_ENGLISH_PKL, 'rb') as f:
            data = pickle.load(f)

        words = data.get('words', set())
        edges = data.get('edges', {})

        print(f"  Words loaded: {len(words):,}")
        print(f"  Edges loaded: {sum(len(v) for v in edges.values()):,}")

        # Test some queries
        test_words = ['apple', 'dog', 'computer', 'love']
        print("\n  Testing sample queries:")
        for word in test_words:
            word_edges = edges.get(word, [])
            print(f"    '{word}': {len(word_edges)} edges")

        print("\n  SUCCESS! ConceptNet offline is ready to use.")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def cleanup_temp_files():
    """Optionally clean up the large compressed file."""
    print("\n" + "=" * 60)
    print("STEP 4: Cleanup (Optional)")
    print("=" * 60)

    if os.path.exists(CONCEPTNET_CSV_GZ):
        size = os.path.getsize(CONCEPTNET_CSV_GZ)
        print(f"The compressed file ({size / 1024 / 1024:.1f} MB) can be deleted")
        print("to save disk space. The pickle file is all you need.")

        response = input("\nDelete compressed file? (y/N): ").strip().lower()
        if response == 'y':
            os.remove(CONCEPTNET_CSV_GZ)
            print("  Deleted!")
        else:
            print("  Keeping file.")


def main():
    """Main setup function."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        CONCEPTNET OFFLINE SETUP                          ║
    ║                                                          ║
    ║  This will download and process ConceptNet data          ║
    ║  for offline use in the Codenames Agent.                 ║
    ║                                                          ║
    ║  Requirements:                                           ║
    ║  - ~2GB disk space during processing                     ║
    ║  - ~100MB final storage                                  ║
    ║  - 10-15 minutes                                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Check if already set up
    if os.path.exists(CONCEPTNET_ENGLISH_PKL):
        size = os.path.getsize(CONCEPTNET_ENGLISH_PKL)
        print(f"ConceptNet offline data already exists ({size / 1024 / 1024:.1f} MB)")
        response = input("Re-run setup? (y/N): ").strip().lower()
        if response != 'y':
            print("\nSetup skipped. Use existing data.")
            return

    # Step 1: Download
    if not download_conceptnet():
        print("\nSetup failed at download step.")
        return

    # Step 2: Process
    if not process_conceptnet():
        print("\nSetup failed at processing step.")
        return

    # Step 3: Verify
    if not verify_setup():
        print("\nSetup verification failed.")
        return

    # Step 4: Cleanup
    cleanup_temp_files()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nYou can now use ConceptNet offline:")
    print("  python src/conceptnet_offline.py")
    print("\nOr import in your code:")
    print("  from src.conceptnet_offline import ConceptNetOffline")
    print("  cn = ConceptNetOffline()")


if __name__ == "__main__":
    main()
