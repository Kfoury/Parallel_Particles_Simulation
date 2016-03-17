make
rm *.out
sbatch job-stampede-gpu
sleep 10
cat *.out
