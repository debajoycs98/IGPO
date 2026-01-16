#!/bin/bash

# 分组管理脚本
GROUP=$1
ACTION=$2
NUM=$3
SCRIPT="tools_server/search_worker.py"
LOG_DIR="./tools_server/logs/$GROUP"
PID_DIR="./tools_server/pids"
DELAY_SEC=90  # 实例启动间隔(秒)

# 确保目录存在
mkdir -p $LOG_DIR $PID_DIR

start_group() {
    echo "Starting group [$GROUP]..."
    for ((i=1; i<=$NUM; i++)); do
        LOG_FILE="$LOG_DIR/worker_$i.log"
        PID_FILE="$PID_DIR/${GROUP}_$i.pid"
        
        # 如果已有进程则跳过
        if [ -f $PID_FILE ] && ps -p $(cat $PID_FILE) > /dev/null; then
            echo "  Instance $i is already running (PID: $(cat $PID_FILE))"
            continue
        fi
        
        nohup python $SCRIPT >> $LOG_FILE 2>&1 &
        echo $! > $PID_FILE
        echo "  Instance $i started (PID: $!)"
        
        # 非最后一个实例则等待
        if [ $i -lt $NUM ]; then
            echo "  Waiting ${DELAY_SEC} seconds before next..."
            sleep $DELAY_SEC
        fi
    done
}

stop_group() {
    echo "Stopping group [$GROUP]..."
    for PID_FILE in $PID_DIR/${GROUP}_*.pid; do
        if [ -f $PID_FILE ]; then
            PID=$(cat $PID_FILE)
            kill -9 $PID && rm $PID_FILE
            echo "  Stopped PID $PID"
        fi
    done
}

case $ACTION in
    start)
        start_group
        ;;
    stop)
        stop_group
        ;;
    restart)
        stop_group
        sleep 1
        start_group
        ;;
    *)
        echo "Usage: $0 <groupname> {start|stop|restart}"
        echo "Example:"
        echo "  $0 web start 3    # Start web group"
        echo "  $0 web stop       # Stop web group"
        echo "  $0 web restart 5 # Restart web group"
        exit 1
esac