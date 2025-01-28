gscp='gcloud compute scp --zone us-central1-a'
echo -e "Getting CG1D"
${gscp} pavi@chflow-intensive:chbank/cg1d/random/cg1d_minwt.tar.gz ./../chbank/cg1d/random/
${gscp} pavi@chflow-intensive:chflow/input/*cg1d_minwt.txt input/
echo -e "Getting NC CPTP"
${gscp} pavi@chflow-intensive:chbank/cptp/regen/nc_cptp_minwt.tar.gz ./../chbank/cptp/regen/wide/
${gscp} pavi@chflow-intensive:chflow/input/*nc_cptp_minwt.txt input/
echo -e "Getting Unitary"
${gscp} pavi@chflow-intensive:chbank/unitary/regen/unitary_minwt.tar.gz ./../chbank/unitary/regen/
${gscp} pavi@chflow-intensive:chflow/input/*unitary_minwt.txt input/
