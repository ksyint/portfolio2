from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import unicodedata

# huggingface 에서 trocr 모델 weight을 가져오고 해당 weight을 fine tuning 하여서 trocr_weight folder에 저장하였습니다. (tokenizer, processor도 같이저장)
# recognize가 받는 이미지는 송장내에서 craft로 크롭된 부분이고 text가 있는곳으로 추정되는 부분입니다.
# 해당 영역에서 있을법한 text내용을 추출합니다. 


def recongize(img):
    processor = TrOCRProcessor.from_pretrained("trocr_weight") 
    model = VisionEncoderDecoderModel.from_pretrained("trocr_weight")
    tokenizer = AutoTokenizer.from_pretrained("trocr_weight")
  
    pixel_values = processor(img, return_tensors="pt").pixel_values 
    generated_ids = model.generate(pixel_values, max_length=64)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = unicodedata.normalize("NFC", generated_text)
    return generated_text
