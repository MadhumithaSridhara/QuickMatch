echo -e "\nEGREP"
rm perl_out CUDA_out serial_out egrep_out
time egrep $1 $2 > egrep_out 
echo -e "\nPERL"
time perl -ne 'print if '/$1/'' $2 > perl_out 
echo -e "\nCUDA"
time ./matchCuda $1 $2 > CUDA_out
#time ./matchCuda $1 $2 > /dev/null
echo -e "\n"
tail -n 5 CUDA_out
echo -e "\n"
time ./nfa $1 $2 > serial_out
wc -l egrep_out
wc -l perl_out
wc -l CUDA_out
wc -l serial_out

