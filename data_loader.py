from datasets import load_dataset, Audio

def dataset_load(aud_path):

    dataset=load_dataset("audiofolder", data_files=[aud_path])
    dataset=dataset.cast_column("audio", Audio(sampling_rate=16000))

    dataset2=dataset['train']

    sample=next(iter(dataset2))
    
    return sample
