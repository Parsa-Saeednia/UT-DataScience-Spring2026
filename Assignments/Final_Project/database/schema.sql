DROP TABLE IF EXISTS performances;
DROP TABLE IF EXISTS pieces;
DROP TABLE IF EXISTS composers;

CREATE TABLE composers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE pieces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    composer_id INT NOT NULL,
    title VARCHAR(500) NOT NULL,
    FOREIGN KEY (composer_id) REFERENCES composers(id) ON DELETE CASCADE,
    UNIQUE(composer_id, title)
);

CREATE TABLE performances (
    id INT AUTO_INCREMENT PRIMARY KEY,
    piece_id INT NOT NULL,
    split VARCHAR(50) NOT NULL,
    year INT NOT NULL,
    duration DOUBLE NOT NULL,
    midi_filename VARCHAR(255) NOT NULL,
    audio_filename VARCHAR(255) NOT NULL,
    FOREIGN KEY (piece_id) REFERENCES pieces(id) ON DELETE CASCADE
);