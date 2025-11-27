#!/bin/bash
# Ensure .yamllint has max line length 80
sed -i '' 's/max: [0-9]*/max: 80/' .yamllint