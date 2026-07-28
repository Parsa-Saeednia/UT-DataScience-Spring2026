import os
import numpy as np
import pandas as pd
import pretty_midi

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
QUANTIZE_RES = 0.125

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['duration'] >= 60.0) & (df['duration'] <= 1000.0)]
    if 'canonical_composer' in df.columns:
        counts = df['canonical_composer'].value_counts()
        viable = counts[counts >= 10].index
        df = df[df['canonical_composer'].isin(viable)]
    return df

def scale_duration(df: pd.DataFrame) -> pd.DataFrame:
    if 'duration' in df.columns and not df.empty:
        train_df = df[df['split'] == 'train'] if 'split' in df.columns else df
        train_df = train_df if not train_df.empty else df
        mean_dur = train_df['duration'].mean()
        std_dur = train_df['duration'].std()
        df['duration_scaled'] = (df['duration'] - mean_dur) / std_dur if std_dur > 0 else 0.0
    return df

def transpose_to_common_key(notes: list) -> list:
    if not notes:
        return []
    pitches = [n.pitch for n in notes]
    pc_counts = np.bincount([p % 12 for p in pitches], minlength=12)
    if np.std(pc_counts) == 0:
        return notes
    maj_corr = [np.corrcoef(pc_counts, np.roll(MAJOR_PROFILE, i))[0, 1] for i in range(12)]
    min_corr = [np.corrcoef(pc_counts, np.roll(MINOR_PROFILE, i))[0, 1] for i in range(12)]
    best_maj_key = np.argmax(maj_corr)
    best_min_key = np.argmax(min_corr)
    if maj_corr[best_maj_key] >= min_corr[best_min_key]:
        shift = -best_maj_key
    else:
        shift = 9 - best_min_key
    if shift > 5: shift -= 12
    elif shift < -6: shift += 12
    for n in notes:
        n.pitch = int(np.clip(n.pitch + shift, 0, 127))
    return notes

def _extract_notes(pm: pretty_midi.PrettyMIDI) -> list:
    notes = sorted([n for inst in pm.instruments for n in inst.notes], key=lambda x: x.start)
    return transpose_to_common_key(notes)

def _build_transition_matrix(pitch_classes: np.ndarray) -> np.ndarray:
    matrix = np.zeros((12, 12))
    for i in range(len(pitch_classes) - 1):
        matrix[pitch_classes[i], pitch_classes[i+1]] += 1
    return matrix

def _build_sequence_tokens(notes: list) -> list:
    tokens = []
    window_vel = []
    for i, n in enumerate(notes):
        raw_step = n.start - notes[i-1].start if i > 0 else 0.0
        raw_dur = n.end - n.start
        quantized_step = int(round(raw_step / QUANTIZE_RES))
        quantized_dur = max(1, int(round(raw_dur / QUANTIZE_RES)))
        window_vel.append(n.velocity)
        if len(window_vel) > 10:
            window_vel.pop(0)
        energy = int(np.clip((sum(window_vel) / len(window_vel)) / 12.7, 1, 10))
        active_pcs = np.zeros(12, dtype=int)
        for prev_n in reversed(notes[max(0, i-50):i]):
            if prev_n.start <= n.start < prev_n.end:
                active_pcs[prev_n.pitch % 12] = 1
        mapped_pitch = int((n.pitch % 36) + 60)
        token = [mapped_pitch, quantized_dur, quantized_step, int(n.velocity), energy]
        token.extend(active_pcs.tolist())
        tokens.append(token)
    return tokens

def _compute_midi_stats(notes: list, total_duration: float) -> dict:
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    intervals = np.abs(np.diff(pitches))
    pitch_classes = pitches % 12
    pc_counts = np.bincount(pitch_classes, minlength=12)
    trans_matrix = _build_transition_matrix(pitch_classes)
    polyphony = np.sum(durations) / total_duration if total_duration > 0 else 0.0
    seq_tokens = _build_sequence_tokens(notes)
    return {
        'notes_per_sec': len(notes) / total_duration if total_duration > 0 else 0.0,
        'avg_pitch': float(np.mean(pitches)),
        'pitch_range': int(np.max(pitches) - np.min(pitches)),
        'avg_velocity': float(np.mean(velocities)),
        'velocity_var': float(np.std(velocities)),
        'avg_note_duration': float(np.mean(durations)),
        'avg_interval': float(np.mean(intervals)),
        'polyphony_rate': float(polyphony),
        'pitch_distribution': (pc_counts / len(notes)).tolist(),
        'transition_matrix': trans_matrix.tolist(),
        'sequence_tokens': seq_tokens
    }

def parse_midi_file(midi_path: str) -> dict:
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = _extract_notes(pm)
        if len(notes) < 50:
            return {}
        return _compute_midi_stats(notes, pm.get_end_time())
    except Exception:
        return {}

def extract_features_batch(df: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    features_list = []
    valid_indices = []
    for idx, row in df.iterrows():
        midi_path = os.path.join(base_dir, row['midi_filename'].replace('/', os.sep))
        if os.path.exists(midi_path):
            features = parse_midi_file(midi_path)
            if features:
                features_list.append(features)
                valid_indices.append(idx)
    features_df = pd.DataFrame(features_list, index=valid_indices)
    return pd.concat([df.loc[valid_indices], features_df], axis=1)

def persist_data(df: pd.DataFrame, pkl_path: str) -> None:
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    df.to_pickle(pkl_path)
    print(f"Feature engineering complete. Saved {len(df)} records to {pkl_path}")

def execute_pipeline() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'maestro-v3.0.0')
    input_pkl = os.path.join(base_dir, 'data', 'dataframes', 'preprocessed_data.pkl')
    output_pkl = os.path.join(base_dir, 'data', 'dataframes', 'final_features.pkl')
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"Input file missing: {input_pkl}")
    df = pd.read_pickle(input_pkl)
    df = filter_outliers(df)
    df = scale_duration(df)
    df = extract_features_batch(df, data_dir)
    persist_data(df, output_pkl)

if __name__ == "__main__":
    execute_pipeline()