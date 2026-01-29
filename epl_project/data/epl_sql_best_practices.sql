-- Example 1:Get latest match results and predictions
SELECT 
    f.home_team, 
    f.away_team, 
    ls.home_goals, 
    ls.away_goals, 
    p.home_win_prob 
FROM fixtures f
LEFT JOIN live_stats ls ON f.fixture_id = ls.fixture_id
LEFT JOIN predictions p ON f.fixture_id = p.fixture_id
WHERE f.status = 'FT'
ORDER BY f.date DESC
LIMIT 5;

-- Example 2: Find Value Bets (where edge > 0.05)
SELECT 
    f.home_team, 
    f.away_team, 
    p.value_bet_side, 
    p.value_bet_edge, 
    o.home_win_odds 
FROM fixtures f
JOIN predictions p ON f.fixture_id = p.fixture_id
JOIN odds o ON f.fixture_id = o.fixture_id
WHERE p.value_bet_edge > 0.05
ORDER BY p.value_bet_edge DESC;
