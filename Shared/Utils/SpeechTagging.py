from datasets import load_dataset

ds = load_dataset("fka/awesome-chatgpt-prompts")


train_ds = ds['train']

for i in range(len(train_ds)):
    print(train_ds[i])