# About
It is a python script, which reduces the time to clean a dataset before it can fed into a model(especially **scikit learn** library), 
because **scikit learn** library don't work well with *missing values* and object ( *categorical variables* ) datatype

# Requirements
you must have install **pandas library** before, if not have yet, 
run `pip install pandas`

# How to use it
> 1. download dataClean.py(script) script
> 2. move script to folder, where dataset is store
> 3. open termial and move to folder 
> 4. run `python dataClean.py <filename | path>`
> 5. the resulted (*cleaned*) file will be back

# Note
* Use this script if your dataset have **rows > 3000**
* supported file conversion are : csv, xlsx, html, json
* While running it asks from you to change the datatype, so read the log carefully
* While running it asks from you to delete a column if it has too many missing value
