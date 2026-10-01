#!/bin/bash

# Run static type checking with ty (Python)
# Outputs filtered results to reports/ folder

set -o pipefail

mkdir -p reports

# Clear cache to ensure consistent results
rm -rf .mypy_cache

echo "============================================"
echo "   Running Python Type Checking (ty)"
echo "============================================"
echo ""

TY_NOISE_REGEX='^INFO '

run_ty() {
  ty check . --verbose "$@" --exclude="examples/playground/**" --exclude="commons-python/**"
}

# Save filtered output to report
run_ty "$@" 2>&1 | grep -Ev "$TY_NOISE_REGEX" > reports/ty.txt
ty_exit_code=${PIPESTATUS[0]}

echo "Python type check complete. Exit code: $ty_exit_code"
echo "Report saved to: reports/ty.txt"
echo ""

# Display filtered output in CLI
run_ty "$@" 2>&1 | grep -Ev "$TY_NOISE_REGEX"

exit $ty_exit_code
