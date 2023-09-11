echo "$(pwd)" >> ./CANF_VC.pth
SITE=$(python3 -c 'import site; print(site.USER_SITE)')
mv ./CANF_VC.pth "$SITE/CANF_VC.pth"

python3 im pip install -r requirements.txt
cd torchac/
python3 setup.py install --user
rm -rf build dist torchac_backend_cpu.egg-info
cd ../
