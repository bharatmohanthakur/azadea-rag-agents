#!/bin/bash
# Monitor OCI ingestion progress — runs every 5 minutes
# Usage: nohup bash monitor_ingestion.sh > /tmp/oci_monitor.log 2>&1 &

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    # Get health
    HEALTH=$(curl -s http://localhost:8074/health 2>/dev/null)
    POINTS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_points',0))" 2>/dev/null)
    ACTIVE=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_jobs',0))" 2>/dev/null)
    
    # Get log stats
    OK=$(grep -c '] OK ' /tmp/oci_batch_v2.log 2>/dev/null || echo 0)
    FAIL=$(grep -c '] FAIL ' /tmp/oci_batch_v2.log 2>/dev/null || echo 0)
    TIMEOUT=$(grep -ci '] timeout' /tmp/oci_batch_v2.log 2>/dev/null || echo 0)
    
    echo "Points: $POINTS | OK=$OK Fail=$FAIL Timeout=$TIMEOUT | Active=$ACTIVE"
    
    # Check if batch is still running
    if ! pgrep -f batch_ingest_parallel > /dev/null 2>&1; then
        echo ""
        echo "=========================================="
        echo "BATCH INGESTION COMPLETE!"
        echo "=========================================="
        echo "Final points: $POINTS"
        echo "OK=$OK Fail=$FAIL Timeout=$TIMEOUT"
        
        # Run comparison test automatically
        echo ""
        echo "Running comparison test..."
        cd /home/admincsp/graphiti_fixed_test/ingestion-oci
        source /home/admincsp/multimodal-rag/azadea/.venv/bin/activate
        python3 test_compare.py 2>&1
        
        echo ""
        echo "DONE — $(date '+%Y-%m-%d %H:%M:%S')"
        break
    fi
    
    sleep 300  # Check every 5 minutes
done
