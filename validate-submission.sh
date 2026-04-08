#!/usr/bin/env bash

set -e

echo "==================================================="
echo " FlowForge AI - Pre-Submission Validation Checklist"
echo "==================================================="

# 1. Check openenv validate
echo ""
echo "▶ Running openenv validator..."
openenv validate --verbose

# 2. Check Inference Script Logging
echo ""
echo "▶ Running inference script..."
export MODEL_NAME="rule-based"
python inference.py > submission_test.log

echo "Checking log formatting..."
if ! grep -q "^\[START\]" submission_test.log; then
    echo "ERROR: [START] tag not found or improperly formatted"
    exit 1
fi
if ! grep -q "^\[STEP\]" submission_test.log; then
    echo "ERROR: [STEP] tag not found or improperly formatted"
    exit 1
fi
if ! grep -q "^\[END\]" submission_test.log; then
    echo "ERROR: [END] tag not found or improperly formatted"
    exit 1
fi

echo "Log format looks correct!"

# 3. Check Docker Build
echo ""
echo "▶ Running Docker build check..."
docker build -t flowforge-ai-test .

# 4. Success
echo ""
echo "==================================================="
echo " ✅ ALL LOCAL CHECKS PASSED "
echo "==================================================="
echo "Note: Full evaluation will also test against missing/invalid endpoints, ping the HF space URL, and impose hardware limits."
