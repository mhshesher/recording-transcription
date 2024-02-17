import os
import torch


def do_transcription(sample, processor, model):

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	model = model.to(device)

	audio_inputs = processor(audios=sample['audio']["array"], sampling_rate=16_000, return_tensors="pt").to(device)
	output_tokens = model.generate(**audio_inputs, tgt_lang="ben", generate_speech=False)
	translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

	return translated_text_from_audio