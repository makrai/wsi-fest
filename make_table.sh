source ~/tool/python/venv3/bin//activate
echo '\begin{tabular}{c|rrr|rrr}'
for exp in `seq 10 16`
do
  voc=`echo 2^$exp|bc`
  echo `python multisense_translate.py --restrict_vocab $voc --prec_level 1  --vanilla-nn-search`;
  echo `python multisense_translate.py --restrict_vocab $voc --prec_level 1`;
  echo `python multisense_translate.py --restrict_vocab $voc --prec_level 5  --vanilla-nn-search`;
  echo `python multisense_translate.py --restrict_vocab $voc --prec_level 5`;
  echo `python multisense_translate.py --restrict_vocab $voc --prec_level 10 --vanilla-nn-search`;
  echo `python multisense_translate.py --restrict_vocab $voc --prec_level 10` \\
  echo '\\'
done
echo '\end{tabular}'
