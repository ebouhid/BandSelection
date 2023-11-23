# Runs experiments for Unet and DeepLabV3+ on all band combinations

# Remove previous .out file and pids.txt
rm nohup.out
rm pids.txt

# Source conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pl

# Function to run a command and save PID and command to pids.txt
run_command_with_pid() {
    command="$1"
    nohup $command >> nohup.out 2>&1 &
    pid=$!
    echo "Command: $command" >> pids.txt
    echo "PID: $pid" >> pids.txt
    wait $pid
}

# Run training scripts sequentially with PID and command printing
run_command_with_pid "python deeplab.py"
run_command_with_pid "python deeplab_pca.py"
run_command_with_pid "python unet.py"
run_command_with_pid "python unet_pca.py"

echo "All done!"
