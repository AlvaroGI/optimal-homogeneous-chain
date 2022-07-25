## For a single case ##
n=3
cutoff=3
echo "Building model n=$n, cutoff=$cutoff"
python valueiter_build.py --n $n --cutoff $cutoff

## For several cases (in parallel, using screens) ##
#for n in 3 4 5 6
#do
#    for cutoff in 2
#    do
#        echo "Building model n=$n, cutoff=$cutoff"
#        screen -S buildmodel_n${n}_p${p}_ps${p_s}_tcut${cutoff} -d -m \
#            python valueiter_build.py --n $n --cutoff $cutoff
#    done
#done