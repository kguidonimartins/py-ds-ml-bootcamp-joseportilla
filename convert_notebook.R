# ipak function: install and load multiple R packages.
# Check to see if packages are installed. 
# Install them if they are not, then load them into the R session.
# Forked from: https://gist.github.com/stevenworthington/3178163
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)){
    install.packages(new.pkg, dependencies = TRUE)
  }
  suppressPackageStartupMessages(sapply(pkg, require, character.only = TRUE))
}

ipak(c("tidyverse", "tidylog", "fs", "pacman"))

pacman::p_load_gh(
"trinker/lexicon",
"trinker/textclean"
)

############################################################
#                                                          #
#                    convert notebooks                     #
#                                                          #
############################################################

# TODO: converter notebook para markdowm e para scripts python
# BODY: ver: https://reproducible-science-curriculum.github.io/publication-RR-Jupyter/02-exporting_the_notebook/index.html

notebook <- 
  fs::dir_info(
  path = "Refactored_Py_DS_ML_Bootcamp-master/",
  recurse = TRUE,
  regexp = "*.ipynb$"
    ) %>% 
  select(path) %>% 
  pull()

system(paste("jupyter nbconvert", paste0("'", notebook[1], "'"), "--to markdown --output", paste0("'", paste0(notebook[1] %>% str_remove("\\..+"), ".md"), "'")), show.output.on.console = TRUE)

cat("jupyter nbconvert", paste0("'", notebook[1], "'"), "--to markdown --output", paste0("'", paste0(notebook[1] %>% str_remove("\\..+"), ".md"), "'"))

      
