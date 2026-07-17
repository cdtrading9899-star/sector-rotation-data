name: Fetch NSE Sector Data

on:
  schedule:
    # 10:45 UTC = 16:15 IST, Mon-Fri — after NSE close (15:30 IST) with a buffer
    - cron: "45 10 * * 1-5"
  workflow_dispatch:   # lets you trigger it manually from the Actions tab, for testing

permissions:
  contents: write       # needed so the workflow can commit the updated JSON back to the repo

jobs:
  fetch:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install yfinance pandas numpy

      - name: Run fetch script
        run: python fetch_sector_data.py

      - name: Commit updated data
        run: |
          git config user.name "sector-data-bot"
          git config user.email "actions@users.noreply.github.com"
          git add sector_data.json
          git diff --quiet --cached || git commit -m "Update sector_data.json ($(date -u +%Y-%m-%d))"
          git push
