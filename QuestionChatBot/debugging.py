from datasets import load_dataset

dataset = load_dataset("json", data_files="QuestionChatBot/dataset/*.jsonl")
print(dataset)
