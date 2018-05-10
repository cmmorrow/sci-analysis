#!/bin/bash
com=$HOME/sci_analysis_qa_env/bin/jupyter
docs=./docs
images=./img
index=sci_analysis_main
notebook=${docs}/${index}.ipynb
output=${docs}/index.md
output_backup=${output}.bak
if [ -f ${output} ]; then
    mv ${output} ${output_backup}
fi
${com} nbconvert --to markdown NbConvertApp.output_base=${docs} --NbConvertApp.output_files_dir=${images} ${notebook}
mv ${docs}/${index}.md ${output}
sed -i -e "s/> Note:/.. note::/g" ${output}
sed -i -e "s/.*warnings.*//" ${output}
make -C ${docs} html
rm ${docs}/*-e

exit 0