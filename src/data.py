from datasets import load_dataset 
from transformers import AutoTokenizer 

def get_dataloaders(model_ckpt, batch_size, max_len):
    dataset = load_dataset("emotion")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


    def tokenize(batch):
        return tokenizer(batch['text'], padding = 'max_length', truncation=True, max_length=max_len)

    dataset = dataset.map(tokenize, batched=True)


    cols = ["input_ids", "attention_mask", "label"]
    dataset.set_format(type="torch", columns=cols)

    from torch.utils.data import DataLoader 

   
    train_loader = DataLoader(dataset['train'], batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset['validation'], batch_size = batch_size)
    test_loader = DataLoader(dataset['test'], batch_size = batch_size)

    return train_loader, val_loader, test_loader, dataset
  

