#!/bin/bash
#SBATCH --job-name=sahil_vs_harshita
#SBATCH --partition=gpuA
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=tournament_sahil_vs_harshita_%j.out

echo "=== SAHIL VS HARSHITA TOURNAMENT (20 GAMES) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

cd ~/scratch/COMP34111-AI-Games-Hex

# Check GPU
nvidia-smi || echo "No GPU detected"

SUMMARY_FILE="tournament_summary_sahil_vs_harshita_${SLURM_JOB_ID}.txt"

# Initialize summary file
cat > "$SUMMARY_FILE" << EOF
================================================================================
                    SAHIL VS HARSHITA TOURNAMENT SUMMARY
================================================================================
Job ID:     $SLURM_JOB_ID
Node:       $(hostname)
Start Time: $(date)
GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "N/A")

AGENTS:
  - SAHIL:    agents.Group12.HexMastUltra2 (HexMastUltra2)
  - HARSHITA: agents.Group12.agent (MCTSAgent)

TOURNAMENT FORMAT: 20 games, alternating who plays RED (first player)
================================================================================

GAME-BY-GAME RESULTS:
---------------------
EOF

SAHIL_WINS=0
HARSHITA_WINS=0
SAHIL_AS_RED_WINS=0
SAHIL_AS_BLUE_WINS=0
HARSHITA_AS_RED_WINS=0
HARSHITA_AS_BLUE_WINS=0
ERRORS=0

for i in {1..20}; do
    GAME_START=$(date +%s)

    if [ $((i % 2)) -eq 1 ]; then
        # Odd games: Sahil RED, Harshita BLUE
        MATCHUP="Sahil (RED) vs Harshita (BLUE)"
        echo "Game $i: $MATCHUP"
        RESULT=$(timeout 600 python3 Hex.py -p1 "agents.Group12.HexMastUltra2 HexMastUltra2" -p2 "agents.Group12.agent MCTSAgent" 2>&1)
    else
        # Even games: Harshita RED, Sahil BLUE
        MATCHUP="Harshita (RED) vs Sahil (BLUE)"
        echo "Game $i: $MATCHUP"
        RESULT=$(timeout 600 python3 Hex.py -p1 "agents.Group12.agent MCTSAgent" -p2 "agents.Group12.HexMastUltra2 HexMastUltra2" 2>&1)
    fi

    GAME_END=$(date +%s)
    GAME_DURATION=$((GAME_END - GAME_START))

    echo "$RESULT" | tail -10

    # Parse winner and update stats
    WINNER="UNKNOWN"
    if echo "$RESULT" | grep -q "RED wins"; then
        if [ $((i % 2)) -eq 1 ]; then
            SAHIL_WINS=$((SAHIL_WINS + 1))
            SAHIL_AS_RED_WINS=$((SAHIL_AS_RED_WINS + 1))
            WINNER="SAHIL"
            echo "  -> SAHIL WINS"
        else
            HARSHITA_WINS=$((HARSHITA_WINS + 1))
            HARSHITA_AS_RED_WINS=$((HARSHITA_AS_RED_WINS + 1))
            WINNER="HARSHITA"
            echo "  -> HARSHITA WINS"
        fi
    elif echo "$RESULT" | grep -q "BLUE wins"; then
        if [ $((i % 2)) -eq 1 ]; then
            HARSHITA_WINS=$((HARSHITA_WINS + 1))
            HARSHITA_AS_BLUE_WINS=$((HARSHITA_AS_BLUE_WINS + 1))
            WINNER="HARSHITA"
            echo "  -> HARSHITA WINS"
        else
            SAHIL_WINS=$((SAHIL_WINS + 1))
            SAHIL_AS_BLUE_WINS=$((SAHIL_AS_BLUE_WINS + 1))
            WINNER="SAHIL"
            echo "  -> SAHIL WINS"
        fi
    else
        ERRORS=$((ERRORS + 1))
        WINNER="ERROR"
        echo "  -> ERROR/UNCLEAR"
    fi

    # Write to summary file
    echo "Game $i: $MATCHUP -> $WINNER (${GAME_DURATION}s)" >> "$SUMMARY_FILE"
    echo ""
done

# Calculate percentages
if [ $SAHIL_WINS -gt 0 ] || [ $HARSHITA_WINS -gt 0 ]; then
    TOTAL_VALID=$((SAHIL_WINS + HARSHITA_WINS))
    SAHIL_PCT=$(echo "scale=1; $SAHIL_WINS * 100 / $TOTAL_VALID" | bc)
    HARSHITA_PCT=$(echo "scale=1; $HARSHITA_WINS * 100 / $TOTAL_VALID" | bc)
else
    SAHIL_PCT="0"
    HARSHITA_PCT="0"
fi

# Append final summary to file
cat >> "$SUMMARY_FILE" << EOF

================================================================================
FINAL RESULTS
================================================================================
SAHIL (HexMastUltra2):     $SAHIL_WINS wins ($SAHIL_PCT%)
  - As RED (first player): $SAHIL_AS_RED_WINS / 10
  - As BLUE (second):      $SAHIL_AS_BLUE_WINS / 10

HARSHITA (MCTSAgent):      $HARSHITA_WINS wins ($HARSHITA_PCT%)
  - As RED (first player): $HARSHITA_AS_RED_WINS / 10
  - As BLUE (second):      $HARSHITA_AS_BLUE_WINS / 10

ERRORS/INVALID GAMES:      $ERRORS

================================================================================
End Time: $(date)
================================================================================
EOF

echo "========================================="
echo "FINAL RESULTS (20 GAMES)"
echo "========================================="
echo "SAHIL (HexMastUltra2):   $SAHIL_WINS wins ($SAHIL_PCT%)"
echo "  - As RED: $SAHIL_AS_RED_WINS/10, As BLUE: $SAHIL_AS_BLUE_WINS/10"
echo "HARSHITA (MCTSAgent):    $HARSHITA_WINS wins ($HARSHITA_PCT%)"
echo "  - As RED: $HARSHITA_AS_RED_WINS/10, As BLUE: $HARSHITA_AS_BLUE_WINS/10"
echo "ERRORS:                  $ERRORS"
echo "========================================="
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo "End time: $(date)"
