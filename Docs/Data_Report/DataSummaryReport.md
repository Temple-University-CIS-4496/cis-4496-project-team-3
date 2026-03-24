# Data Report
This file will be generated for each data file received or processed. The Interactive Data Exploration, Analysis, and Reporting (IDEAR) utility developed by TDSP team of Microsoft can help you explore and visualize the data in an interactive way, and generate the data report along with the process of exploration and visualization. 

IDEAR allows you to output the data summary, statistics, and charts that you want to use to tell the data story into the report. You only need to click a few buttons, and the report will be generated for you. 

## General summary of the data

## Data quality summary

We wrote a script that scans a folder of script text files, finds exact duplicates and near duplicates, and sorts them into a new output directory. It starts by collecting all .txt files while skipping folders that already belong to previous output. Each file is read, normalized, and tokenized so that small formatting differences do not affect comparison. For every script, the program computes an exact content fingerprint using SHA256 and also computes a simhash value, which gives a compact way to measure approximate text similarity. It also extracts title-like tokens from the filename so the script can later compare names in a structured way.

The grouping happens in two stages. First, files with identical SHA256 hashes are placed into exact duplicate buckets. Then one representative from each exact bucket is compared to the others using simhash distance and token count ratio, which allows the script to detect near duplicates with similar content. Since content similarity alone can incorrectly merge unrelated scripts, the program refines those candidate groups by checking filename title similarity and splitting groups whose names are too different. After that, each final group is labeled as unique, exact duplicate, or near duplicate. The script copies unique files into a unique folder, copies duplicate groups into numbered folders, and also picks one canonical file from each duplicate group to place in the unique folder as its representative. At the end, it writes JSON manifest and summary files so the result can be checked later.

Total files: 8,005 

Unique files: 4,225 (52.78%) 
Exact duplicates: 306 (3.82%) 
Near duplicates: 3,474 (43.40%) 

Exact groups: 143 
Near groups: 600

Avg exact group size: 2.14 
Avg near group size: 5.79

## Target variable

## Individual variables

## Variable ranking

## Relationship between explanatory variables and target variable


