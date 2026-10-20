#!/bin/bash
#SBATCH -N 1                # 1 node
#SBATCH -J ESMRank
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4

echo ""
echo "   _____ ________  _________            _ "
echo "  |  ___/  ___|  \/  || ___ \          | | "
echo "  | |__ \ \`--.| .  . || |_/ /__ _ _ __ | | __"
echo "  |  __| \`--. \ |\/| ||    // _\` | '_ \| |/ /"
echo "  | |___/\__/ / |  | || |\ \ (_| | | | |   <  "
echo "  \____/\____/\_|  |_/\_| \_\__,_|_| |_|_|\_\  "
echo ""

# Thin wrapper kept for SLURM / Slivka: all the logic lives in the `esmrank` package.
# Usage: run_esmrank_pipeline.sh [-h] --input INPUT [-csv] [-sub] [-indel] [-alanines]
if command -v esmrank >/dev/null 2>&1; then
	esmrank run "$@"
else
	# not pip-installed: run straight from the repository checkout
	PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m esmrank run "$@"
fi
