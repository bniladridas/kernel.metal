#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 KERNEL.METAL (harpertoken)

# Commit message validation script
# Checks if commit message follows conventional commit format

commit_msg_file=$1
commit_msg=$(cat "$commit_msg_file")

# Regex for conventional commit: type(scope): description
# Types: feat, fix, docs, ci, chore, refactor, test, style, perf
if ! echo "$commit_msg" | head -n1 | grep -qE '^(feat|fix|docs|ci|chore|refactor|test|style|perf)(\(.+\))?: .+'; then
    echo "❌ Commit message title does not follow conventional commit format."
    echo "Expected: type(scope): description"
    echo "Examples: feat: add new feature"
    echo "          fix(ui): resolve button issue"
    echo "          docs: update readme"
    exit 1
fi

# Check for body (at least one additional line)
if [ $(echo "$commit_msg" | wc -l) -lt 2 ]; then
    echo "❌ Commit message must include a body with details."
    echo "Example:"
    echo "feat: add new feature"
    echo ""
    echo "add bounds check and update docs"
    exit 1
fi

# Check for lowercase (no uppercase letters)
if echo "$commit_msg" | grep -q '[A-Z]'; then
    echo "❌ Commit message must be in lowercase."
    echo "Avoid uppercase letters (e.g., use 'add' not 'Add')."
    exit 1
fi

echo "✅ Commit message format is valid."