import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class Wav2Vec(nn.Module):
    def __init__(self, model_path:str, device:str)-> None:
        super(Wav2Vec, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).cuda()
        self.device = device
    def forward(self, audios: torch.Tensor)-> torch.Tensor:
        input_values = []
        attention_masks = []
        for audio in audios:
            feature = self.processor(audio,
                        sampling_rate=self.processor.feature_extractor.sampling_rate, 
                        return_tensors="pt", padding="longest")
            input_values.append(feature.input_values)
            attention_masks.append(feature.attention_mask)
        input_values = torch.concat(input_values, dim=0).cuda()
        attention_masks = torch.concat(attention_masks, dim=0).cuda()
        logits = self.model(input_values, attention_mask=attention_masks).logits 
        return logits