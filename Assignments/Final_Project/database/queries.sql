USE maestro_db;

SELECT c.name AS composer, p.title AS piece_title, pf.year, pf.duration 
FROM performances pf
JOIN pieces p ON pf.piece_id = p.id
JOIN composers c ON p.composer_id = c.id
LIMIT 10;

SELECT c.name AS composer, COUNT(pf.id) AS total_performances 
FROM performances pf
JOIN pieces p ON pf.piece_id = p.id
JOIN composers c ON p.composer_id = c.id
GROUP BY c.id 
ORDER BY total_performances DESC 
LIMIT 5;

SELECT pf.year, ROUND(AVG(pf.duration), 2) AS avg_duration_seconds 
FROM performances pf
GROUP BY pf.year 
ORDER BY pf.year ASC;

SELECT c.name AS composer, p.title, pf.duration 
FROM performances pf
JOIN pieces p ON pf.piece_id = p.id
JOIN composers c ON p.composer_id = c.id
ORDER BY pf.duration DESC 
LIMIT 1;

SELECT split, COUNT(*) AS count 
FROM performances 
GROUP BY split;