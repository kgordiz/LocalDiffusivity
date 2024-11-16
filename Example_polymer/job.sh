#!/bin/bash
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --constraint=centos7
#SBATCH --mem=186G
#SBATCH --ntasks-per-node=40
#SBATCH -p sched_mit_ase
#SBATCH -o C.out 
#SBATCH -e C.err
#SBATCH -J Mg

module purge
module load vasp/ase/5.4.4

echo "$SLURM_NTASKS slurm tasks allocated"

#cd ${PBS_O_WORKDIR}

address=$(pwd)
echo $address
for (( i=1; i<=100; i++ ))
  do
  mkdir disp$i
  cd disp$i
  name=$i
  cp $address/all_random_disps/POSCAR$name ./POSCAR
  cp $address/needed_vasp_files/INCAR .
  cp $address/needed_vasp_files/POTCAR .
  cp $address/needed_vasp_files/KPOINTS .
  mpirun -np $SLURM_NTASKS vasp_std > out.txt
  cd $address
done

directory_name='xml_collection'
mkdir $directory_name
cd $directory_name

for (( i=1; i<=100; i++ ))
  do
    cp ../disp$i/vasprun.xml ./vasprun$i.xml
  done
cd $address/
