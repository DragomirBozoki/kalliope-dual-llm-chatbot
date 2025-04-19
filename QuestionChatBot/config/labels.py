# labels.py

label2id = {
    "lab-on": 0,
    "lab-off": 1,
    "Meeting-on": 2,
    "Meeting-off": 3,
    "kitchen-on": 4,
    "kitchen-off": 5,
    "Livingroom-on": 6,
    "Livingroom-off": 7,
    "Livingroom-dim": 8,
    "room-on": 9,
    "room-off": 10,
    "reading-on": 11,
    "reading-off": 12,
    "ambient-random": 13,
    "ambient-stop": 14,
    "ambient-specific": 15,
    "ambient-sleep": 16,
    "find-my-phone": 17,
    "findkeys": 18,
    "run-web-radio": 19,
    "run-web-radio2": 20,
    "stop-web-radio-stop-web-radio2": 21,
    "exting": 22,
    "check-email": 23,
    "news-sport": 24,
    "run-web-esda": 25,
    "close-web-esda": 26,
    "goodbye": 27,
    "dinner": 28,
    "apartment": 29,
    "sonos-play": 30,
    "sonos-stop": 31,
    "fan-on": 32,
    "fan-off": 33,
    "door-on": 34,
    "Temperature-set": 35,
    "fan-lab1": 36,
    "fan-lab2": 37,
    "room-on1": 38,
    "room-off2": 39,
    "kitchen-on1": 40,
    "kitchen-off1": 41,
    "saytemp": 42,
    "get-the-weather": 43,
    "say-local-date": 44,
    "say-local-date-from-template": 45,
    "tea-time": 46,
    "remember-synapse": 47,
    "remember-todo": 48,
}

# Reverse mapping for inference
id2label = {v: k for k, v in label2id.items()}

reminder_keywords = [
    # English
    "remind me to", "i have to take", "in", "tonight",
    "in a few", "in x", "seconds", "minutes", "hours", "days",

    # Greek put out please let me know
    "υπενθύμισέ μου να", "πρέπει να", "σε", "σε λίγα",
    "λεπτά", "ώρες", "ημέρες", "παρακαλώ να με ενημερώσεις", "να πάρω", "να θυμηθώ"
]
