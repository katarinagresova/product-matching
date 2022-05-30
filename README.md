# Product Matching

## Installation

You can install this package using pip and github repository as follows:

```bash
pip install git+https://github.com/katarinagresova/product-matching
```

## Local development

If you want to run experiments from this repository or contribute to the package, use following commands to clone the repository and install the package into virtual environment.

```bash
git clone git@github.com:katarinagresova/product-matching.git
cd product-matching

virtualenv venv --python=python3.8
source venv/bin/activate

pip install -e .
```

## Usage

Matching model with default settings can be executed with four lines of code:


```python
from matching.model.model import MatchingModel

model = MatchingModel()
model.fit(client_data, client_name, seller_data, seller_name)
matches = model.predict()
```

Where `client_data` and `seller_data` are pandas DataFrames with client and seller data (format is described below). Matching pairs always contain one product from client and second product is found in products from seller. `client_name` and `seller_name` are names of the client and seller respectively and are used to mark columns in the output `matches` DataFrame.

## Advanced usage

There are several points of customization of the model. 

### Model components

When creating a model, you can specify the following parameters:

```python
from matching.model.model import MatchingModel
from matching.model.tokenizer import words
from matching.model.transform import tfidf
from matching.model.similarity import cossim_top
from matching.model.matcher import best_unique

model = MatchingModel(
    tokenizer=words, 
    transform=tfidf, 
    similarity=cossim_top, 
    matcher=best_unique
)
```

Presented arguments are the default ones and are implemented in a `matching` package. You can also use your own tokenizer, transform, similarity and matcher, they must be specified as functions with following functionality:

- `tokenizer`: a function that takes a string as an input, and splits it into list of string tokens. 
- `transform`: a function that takes two pd.Series of strings and tokenizer as inputs, and both pd.Series converts into sparse matrices.
- `similarity`: a function that takes two sparse matrices (and some additional parameters), and computes similarity between all rows from first matrix and all rows from second matrix. Results are also returned as sparse matrix.
- `matcher`: a function that takes sparse matrix of similarities and two pd.Series of ids of input data, and returns best matching pairs and their matching score.

### Predicting

All arguments of `model.fit()` are compulsory, but `model.predict()` allows for additional configuration. Following arguments are available:

```python
from matching.model.model import MatchingModel

model = MatchingModel()
model.fit(client_data, client_name, seller_data, seller_name)

matches = model.predict(
    lower_bound=0.7, 
    do_ean_check=True,
    exact_match_columns=[column_one, column_two], 
    exact_match_columns_with_nan=[column_three, column_four, column_five],
)
```

- `lower_bound`: a float value between 0 and 1, that is used to filter out matches with similarity score lower than this value.
- `do_ean_check`: a boolean value, if True, then matching of products with different EAN codes is not allowed.
- `exact_match_columns`: a list of columns that are used to split products into groups with the same values of these columns.
- `exact_match_columns_with_nan`: a list of columns that are also used to split products into groups with the same values of these columns, but also allowing for NaN values.


## Data format

Matching model is expecting data in specific format. Keys of dictionary `data` are names of sellers and values are pd.DataFrame with four columns: 

Required columns:
- **id**: unique identifier of product within one seller.
- **target**: text information about product.

Specific not required columns:
- **ean**: unique identifier of product across all sellers.

Additional match columns:
- **exact_match_columns**: columns that are used to split products into groups with the same values of these columns. Passed into `exact_match_columns` argument of `model.predict()` described above.
- **exact_match_columns_with_nan**: columns that are used to split products into groups with the same values of these columns, but also allowing for NaN values. Passed into `exact_match_columns_with_nan` argument of `model.predict()` described above.

### Example

| id   |      ean      |  group_id | target | speed_index | load_index |
|:----------:|:-------------:|:------:|:------:|:------:|:------:|
| https://www.pneuboss.cz/pneu-235-50-r-19-99y-sportcontact_6-tl-zr-fr-mo1-contin |  4019238006629 | CONTINENTAL 50 235 19 Y 99 | CONTINENTAL SPORTCONTACT 6 235/50 R 19 99Y CAR SPORTCONTACT 6 SUMMER R TL,FR,ZR C A B 3952 CONTINENTAL 50 235 19 Y 99 | Y | 99 |
| https://www.pneuboss.cz/pneu-165-70-r-14-85t-blizzak_lm005-tl-xl-m-s-3pmsf-brid | | BRIDGESTONE 70 165 14 T 85 | BRIDGESTONE BLIZZAK LM005 165/70 R 14 85T CAR BLIZZAK LM005 WINTER R 3PMSF,TL,M+S,XL C A B 3031 BRIDGESTONE 70 165 14 T 85 | T | 85 |

In this example, `exact_match_columns = ['group_id']` and `exact_match_columns_with_nan = ['speed_index', 'load_index']`.