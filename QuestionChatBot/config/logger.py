import csv

def log_entry(user_input, corrected_input, intent_label, intent_prob, gen_response):
    with open("chat_log.csv", mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([user_input, corrected_input, intent_label, intent_prob, gen_response])
