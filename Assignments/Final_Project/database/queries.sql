SELECT
    c.name AS canonical_composer,
    p.title AS canonical_title,
    pf.split,
    pf.year,
    pf.duration,
    pf.midi_filename,
    pf.audio_filename
FROM performances pf
JOIN pieces p ON pf.piece_id = p.id
JOIN composers c ON p.composer_id = c.id;