import os
import torch
import pretty_midi
import numpy as np
from train import MusicLSTM
from feature_engineering import parse_midi_file


OUTPUT_FILE_NAME = 'music3.mid'
INPUT_FILE_NAME = 'MIDI-Unprocessed_02_R3_2008_01-03_ORIG_MID--AUDIO_02_R3_2008_wav--3.midi'
FOLDER_NAME = '2008'

def apply_repetition_penalty(logits, generated_pitches, penalty=1.2):
    if not generated_pitches:
        return logits
    recent_pitches = generated_pitches[-10:]
    for pitch in set(recent_pitches):
        class_idx = pitch - 60
        if 0 <= class_idx < 36:
            if logits[0, class_idx] > 0:
                logits[0, class_idx] /= penalty
            else:
                logits[0, class_idx] *= penalty
    return logits

def sample_top_k(logits, k=5, temperature=0.85):
    logits = logits / temperature
    values, indices = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return indices[0, sampled_idx[0, 0]].item()

def build_dynamic_harmonic_context(generated_pitches):
    active_pcs = np.zeros(12, dtype=float)
    if not generated_pitches:
        return active_pcs
    recent_pitches = generated_pitches[-8:]
    for p in recent_pitches:
        active_pcs[p % 12] = 1.0
    return active_pcs

def generate_continuation_for_trackX():
    print("\n--- Starting MIDI Prompt & Continuation ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'project' else current_dir
    
    selected_midi_rel_path = os.path.join('data', 'maestro-v3.0.0', FOLDER_NAME , INPUT_FILE_NAME)
    midi_path = os.path.join(base_dir, selected_midi_rel_path)
    model_path = os.path.join(base_dir, 'models', 'music_lstm.pth') 
    output_path = os.path.join(base_dir, 'output', OUTPUT_FILE_NAME)
    
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"Selected MIDI file not found at: {midi_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MusicLSTM(input_dim=17, hidden_dim=128, num_layers=3, vocab_size=36)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Processing target file:\n -> {selected_midi_rel_path}")
    features = parse_midi_file(midi_path)
    
    if not features or 'sequence_tokens' not in features or len(features['sequence_tokens']) < 50:
        raise ValueError("Selected MIDI file contains fewer than 50 valid extracted tokens.")
        
    full_tokens = features['sequence_tokens']
    
    cut_seed_tokens = np.array(full_tokens[:50])
    current_seq = torch.tensor(cut_seed_tokens, dtype=torch.float32).unsqueeze(0).to(device)
    
    generated_pitches = [int(token[0]) for token in cut_seed_tokens]
    
    num_notes_to_predict = 100
    print(f"Predicting the next {num_notes_to_predict} upcoming notes...")
    
    with torch.no_grad():
        for _ in range(num_notes_to_predict):
            output = model(current_seq)
            
            output = apply_repetition_penalty(output, generated_pitches, penalty=1.2)
            predicted_class = sample_top_k(output, k=5, temperature=0.85)
            
            mapped_pitch = predicted_class + 60
            generated_pitches.append(mapped_pitch)
            
            next_token = np.zeros(17)
            next_token[0] = mapped_pitch
            next_token[1] = 2   
            next_token[2] = 2   
            next_token[3] = 80  
            next_token[4] = 5   
            
            harmonic_context = build_dynamic_harmonic_context(generated_pitches)
            next_token[5:17] = harmonic_context
            
            next_token_tensor = torch.tensor(next_token, dtype=torch.float32).view(1, 1, 17).to(device)
            current_seq = torch.cat((current_seq[:, 1:, :], next_token_tensor), dim=1)
            
    print("Writing prompt + predicted notes to MIDI...")
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  
    
    QUANTIZE_RES = 0.125  
    current_time = 0.0
    
    for pitch in generated_pitches:
        step_time = 2 * QUANTIZE_RES 
        dur_time = 2 * QUANTIZE_RES
        
        current_time += step_time
        safe_pitch = int(np.clip(pitch, 0, 127))
        
        note = pretty_midi.Note(
            velocity=80, 
            pitch=safe_pitch, 
            start=current_time, 
            end=current_time + dur_time
        )
        piano.notes.append(note)
        
    pm.instruments.append(piano)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pm.write(output_path)
    print(f"\nExecution Complete! Generated file saved to:\n -> {output_path}")

if __name__ == "__main__":
    generate_continuation_for_trackX()