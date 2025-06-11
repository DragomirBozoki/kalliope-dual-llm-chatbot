import os
import csv

# LOG response header, creates the file if it doesn't exist
def init_log_file():
    log_file = "chat_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_input", "corrected_input", "intent_label", "intent_prob", "gen_response"])

