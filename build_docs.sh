#!/bin/bash

# The path to the jupyter executable.
com=jupyter

# The path to the docs directory.
docs=./docs

# The path to the documentation image directory under docs.
images=img

# The path to the toc file. This file is combined with the welcome page.
toc=${docs}/toc.rst

# The path to the README file.
readme=./README

# The filename of the welcome page. This will be a combination of the README and toc file.
index=${docs}/index.rst

#index=sci_analysis_main

# The list of notebooks to be converted to md for documentation.
notebooks=(
    getting_started
    using_sci_analysis
    pandas
    distribution
    frequency
    bivariate
    location_test
)
for nb in ${notebooks[*]}; do
    notebook=${docs}/${nb}.ipynb
    output=${docs}/${nb}.md
    output_backup=${output}.bak
    if [[ -f ${output} ]]; then
        mv ${output} ${output_backup}
    fi
    ${com} nbconvert --to markdown NbConvertApp.output_base=${docs} --NbConvertApp.output_files_dir=${images} ${notebook}
    sed -i -e "s/> Note:/.. note::/g" ${output}
    sed -i -e "s/.*warnings.*//" ${output}
done

m2r ${readme}.md
cat ${readme}.rst ${toc} > ${index}
make -C ${docs} html
rm ${readme}.rst
rm ${docs}/*-e

exit 0
