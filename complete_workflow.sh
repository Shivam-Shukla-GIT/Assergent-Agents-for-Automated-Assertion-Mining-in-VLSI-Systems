#!/bin/bash
# complete_workflow.sh - Automated Classification, Split, and Database Generation
set -e  # Exit on error
echo "=========================================="
echo "RAG ASSERTION FRAMEWORK - COMPLETE SETUP"
echo "=========================================="

# Configuration
INPUT_DIR="database"           # Directory with assertion files
OUTPUT_DIR="processed"
CLASSIFICATION_FILE="$OUTPUT_DIR/classification_report.json"
KNOWLEDGE_DB="$OUTPUT_DIR/knowledge_db"
TEST_SET="$OUTPUT_DIR/test_set"
FINAL_DB="specs_database.json"
INTERACTIVE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --interactive|-i)
      INTERACTIVE=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--input INPUT_DIR] [--output-dir OUTPUT_DIR] [--interactive|-i]"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 1: Classification
echo ""
echo "Step 1/3: Classifying assertions..."
echo "======================================"
python3 assertion_classifier.py --input "$INPUT_DIR" --output "$CLASSIFICATION_FILE"

# Count total files
TOTAL_FILES=$(grep -o '"total_assertions":' "$CLASSIFICATION_FILE" | wc -l)
if [ $TOTAL_FILES -eq 0 ]; then
  TOTAL_FILES=$(grep -o '"spec_id"' "$CLASSIFICATION_FILE" | wc -l)
fi
if [ $TOTAL_FILES -eq 0 ]; then
  # Try to count files directly
  TOTAL_FILES=$(find "$INPUT_DIR" -type f -name "*.txt" | wc -l)
fi

# Calculate default train/test sizes (2/3 - 1/3)
if [ $TOTAL_FILES -gt 0 ]; then
  TRAIN_SIZE=$(($TOTAL_FILES * 2 / 3))
  TEST_SIZE=$(($TOTAL_FILES - $TRAIN_SIZE))
else
  # Default if we couldn't determine
  TRAIN_SIZE=20
  TEST_SIZE=10
fi

# Step 2: Stratified Split
echo ""
echo "Step 2/3: Stratified train/test split..."
echo "========================================="

# Set interactive flag if requested
INTERACTIVE_FLAG=""
if [ "$INTERACTIVE" -eq 1 ]; then
  INTERACTIVE_FLAG="--interactive"
fi

# Run splitter with automatic defaults (2/3 train, 1/3 test)
python3 stratified_splitter.py \
  --classification "$CLASSIFICATION_FILE" \
  --input "$INPUT_DIR" \
  --output-train "$KNOWLEDGE_DB" \
  --output-test "$TEST_SET" \
  --seed 42 \
  $INTERACTIVE_FLAG

# Step 3: Generate Database
echo ""
echo "Step 3/3: Generating JSON database..."
echo "======================================"
python3 prepare_database.py --input "$KNOWLEDGE_DB" --output "$FINAL_DB" --classification "$OUTPUT_DIR/split_metadata.json"

# Count actual files in each directory
TRAIN_COUNT=$(find "$KNOWLEDGE_DB" -maxdepth 1 -type f -name "*.txt" | wc -l)
TEST_COUNT=$(find "$TEST_SET" -maxdepth 1 -type f -name "*.txt" | wc -l)
TOTAL_COUNT=$((TRAIN_COUNT + TEST_COUNT))

# Clear directories before copying if they exist
if [[ "$CLEAN_DIRS" -eq 1 ]]; then
  rm -f "$KNOWLEDGE_DB"/*.txt
  rm -f "$TEST_SET"/*.txt
fi

echo ""
echo "=========================================="
echo "COMPLETE! Files generated:"
echo "=========================================="
echo "1. Classification: $CLASSIFICATION_FILE"
echo "2. Knowledge DB:   $KNOWLEDGE_DB/ ($TRAIN_COUNT files)"
echo "3. Test Set:       $TEST_SET/ ($TEST_COUNT files)"
echo "4. Final Database: $FINAL_DB"
echo "5. Split Metadata: $OUTPUT_DIR/split_metadata.json"
echo ""
echo "Split Distribution:"
echo "- Training: $TRAIN_COUNT files ($(echo "scale=1; $TRAIN_COUNT*100/$TOTAL_COUNT" | bc)%)"
echo "- Testing:  $TEST_COUNT files ($(echo "scale=1; $TEST_COUNT*100/$TOTAL_COUNT" | bc)%)"
