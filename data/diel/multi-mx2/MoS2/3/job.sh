#PBS -N MoS2-3
#PBS -r n
#PBS -q main
#PBS -l nodes=28,walltime=12:00:00
cmd=$VASP_SCRIPT
relax_path=$PBS_O_WORKDIR/relax		# must use to change dir
eps_path=$PBS_O_WORKDIR/eps		# must use to change dir

# relax part
echo "Calculating relaxation now"
cd $relax_path
if [[ ! -f CONTCAR.RELAXED ]] || [[ -s CONTCAR.RELAXED ]]
then
    # Copy CONTCAR if last run not finished
    if ! [[ -s CONTCAR ]]
    then
	cp CONTCAR POSCAR
    fi
    $cmd
    if (( $? == 0 ))
    then
	cp CONTCAR CONTCAR.RELAXED
	cp CONTCAR $relax_path/POSCAR
	echo "Relax finished"
    fi
else
    echo "System relaxed, use directly"
    cp CONTCAR.RELAXED $eps_path/POSCAR
fi

# Dielectric part
echo "Calculating dielectrics"
cd $eps_path
if [[ ! -f OUTCAR.COMPLETE ]] || [[ -s OUTCAR.COMPLETE ]]
then
    # Copy CONTCAR if last run not finished
    if [[ ! -f POSCAR ]]
    then
	cp $relax_path/CONTCAR POSCAR
    fi
    $cmd
    if (( $? == 0 ))
    then
	cp OUTCAR OUTCAR.COMPLETE
	cp vasprun.xml vasprun.xml.complete
	echo "Eps finished"
    fi
else
    echo "Eps finished"
fi
