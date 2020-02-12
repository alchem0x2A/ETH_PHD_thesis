#!/bin/bash
for m in Mo W
do
    for x in S Se Te
    do
	for i in 1 2 3 4 5 6
	do
	    if [[ -f ${m}${x}2/${i}/eps/OUTCAR.COMPLETE ]]
	    then
		echo "Finished " ${m}${x}2 $i
	    fi
	done
    done
done
      

		
