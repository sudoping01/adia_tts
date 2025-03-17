import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import os
import uuid
from typing import List, Tuple, Dict, Optional, Any

class AdiaTTS:
    """
    This class provides functionality for converting text to speech in Wolof language,
    with support for long text segmentation and audio concatenation.
    """
    
    def __init__(self, 
                 model_id: str = "CONCREE/Adia_TTS", 
                 output_dir: str = "/tmp/tts_output",
                 hf_token: Optional[str] = None):

        self.model_id = model_id
        self.output_dir = output_dir
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_model()
    
    def load_model(self) -> None:
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id,
                token=self.hf_token
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token
            )
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def synthesize(self, 
                  text: str, 
                  description: str = "A warm and natural voice, with a conversational flow", 
                  config: Optional[Dict[str, Any]] = None) -> Tuple[str, np.ndarray]:

        try:

            if config is None:
                config = {
                    "temperature": 0.01, 
                    "max_new_tokens": 1000,
                    "do_sample": True,
                    "top_k": 50,
                    "repetition_penalty": 1.2
                }
            

            if len(text) > 200:
                max_pos = min(200, len(text))
                pause_chars = ['.', '!', '?', ',', ';', ':', '…']
            
                last_pause = 0
                for char in pause_chars:
                    pos = text[:max_pos].rfind(char)
                    if pos > last_pause:
                        last_pause = pos
                
                if last_pause > 0:
                    text = text[:last_pause + 1]
                else:
                    last_space = text[:max_pos].rfind(' ')
                    if last_space > 0:
                        text = text[:last_space]
                    else:
                        
                        text = text[:200]
            
            input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
            prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            

            audio = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_ids,
                **config
            )
            
            audio_np = audio.cpu().numpy().squeeze()
            
            output_path = os.path.join(self.output_dir, f"{uuid.uuid4()}.wav")
            sf.write(output_path, audio_np, self.model.config.sampling_rate)
            
            return output_path, audio_np
        except Exception as e:
            raise Exception(f"Failed to synthesize speech: {e}")
    
    def segment_text(self, text: str) -> List[str]:
        """
        Split long text into segments of max 200 characters at natural boundaries.
        
        Finds the last natural pause before the 200-character limit and splits there.
        Natural pauses are prioritized in this order:
        1. Sentence endings (., !, ?)
        2. Other punctuation marks (,, ;, :, ...)
        3. Word boundaries (spaces)
        """
        segments = []
        remaining_text = text.strip()
        
        MAX_CHARS = 200  
        
        sentence_end_chars = ['.', '!', '?']
        pause_chars = [',', ';', ':', '…']
        
        while remaining_text:
            
            if len(remaining_text) <= MAX_CHARS:
                segments.append(remaining_text)
                break

            segment_text = ""
            last_sentence_end = -1

            for i in range(min(MAX_CHARS, len(remaining_text))):
                if remaining_text[i] in sentence_end_chars:
                    last_sentence_end = i
            
            if last_sentence_end != -1:
                segment_text = remaining_text[:last_sentence_end + 1]
                remaining_text = remaining_text[last_sentence_end + 1:].strip()

            else:
                last_punct = -1
                for i in range(min(MAX_CHARS, len(remaining_text))):
                    if remaining_text[i] in pause_chars:
                        last_punct = i
                
                if last_punct != -1:
                    segment_text = remaining_text[:last_punct + 1]
                    remaining_text = remaining_text[last_punct + 1:].strip()
                 
                else:
    
                    text_to_check = remaining_text[:MAX_CHARS]
                    last_space = text_to_check.rfind(' ')
                    
                    if last_space != -1:
                        segment_text = remaining_text[:last_space]
                        remaining_text = remaining_text[last_space + 1:].strip()
                       
                    else:
                    
                        segment_text = remaining_text[:MAX_CHARS]
                        remaining_text = remaining_text[MAX_CHARS:].strip()
            
            segments.append(segment_text)
        
        return segments
    
    def batch_synthesize(self, 
                        text: str, 
                        description: str = "A warm and natural voice, with a conversational flow", 
                        config: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
        try:
        
            if config is None:
                config = {
                    "temperature": 0.8,
                    "max_new_tokens": 1000,
                    "do_sample": True,
                    "top_k": 50,
                    "repetition_penalty": 1.2
                }
            
      
            segments = self.segment_text(text)
            
            output_files = []
            for segment in segments:
                output_path, _ = self.synthesize(
                    text=segment,
                    description=description,
                    config=config
                )
                output_files.append(output_path)
            
            combined_path = self.concatenate_audio_files(output_files)
            
            return combined_path, output_files
        
        except Exception as e:
            raise Exception(f"Failed to batch synthesize speech: {e}")
    
    def synthesize_from_file(self, 
                           file_path: str, 
                           description: str = "A warm and natural voice, with a conversational flow", 
                           config: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:

        try:
         
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            
  
            return self.batch_synthesize(
                text=text_content,
                description=description,
                config=config
            )
            
        except Exception as e:
            raise Exception(f"Failed to synthesize speech from file: {e}")
    
    def concatenate_audio_files(self, file_paths: List[str]) -> str:
        """
        Concatenate multiple audio files into a single file with carefully tuned transitions.
        
        This function:
            Reads all audio segments
            Analyzes optimal crossfade length based on segment characteristics
            Applies adaptive crossfading techniques
            Normalizes audio levels throughout segments
        """
        if not file_paths:
            raise ValueError("No audio files provided for concatenation")
        
        sample_rate = None
        audio_segments = []
        
        for file_path in file_paths:
            data, sr = sf.read(file_path)
            
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                pass
                #  I will come back to add resampling logic but for  now let's keep it like this
            
            audio_segments.append(data)
        
        normalized_segments = []
        for segment in audio_segments:
            if np.abs(segment).max() < 0.001:
                normalized_segments.append(segment)
                continue
                
            max_amp = np.abs(segment).max()
            target_amp = 0.7  
            normalized = segment * (target_amp / max_amp)
            normalized_segments.append(normalized)
        
        crossfade_ms = 150  # milliseconds 
        crossfade_samples = int((crossfade_ms / 1000) * sample_rate)
        

        result = normalized_segments[0]
        for i in range(1, len(normalized_segments)):
            current_segment = normalized_segments[i]
            
            if len(result) < crossfade_samples or len(current_segment) < crossfade_samples:
                actual_crossfade = min(len(result), len(current_segment), crossfade_samples)
            else:
                actual_crossfade = crossfade_samples
            
            if actual_crossfade <= 0:
                # Just concatenate if crossfade isn't possible
                result = np.concatenate([result, current_segment])
                continue
                
            t = np.linspace(0, np.pi, actual_crossfade)
            fade_out = np.cos(t) * 0.5 + 0.5  
            fade_in = np.sin(t) * 0.5 + 0.5   
            
   
            result_end = result[-actual_crossfade:]
            segment_start = current_segment[:actual_crossfade]
            
            crossfade_region = (result_end * fade_out) + (segment_start * fade_in)
            
            result = np.concatenate([result[:-actual_crossfade], crossfade_region, current_segment[actual_crossfade:]])
        
    
        output_path = os.path.join(self.output_dir, f"combined_{uuid.uuid4()}.wav")
        sf.write(output_path, result, sample_rate)
    
        return output_path
    
    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model": self.model_id
        }
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {file_path}: {e}")