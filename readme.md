# Context matters in emergency disease control

Install using:

```bash
git clone git@github.com:p-robot/context_matters.git
cd context_matters
python setup.py install --user
```

Testing can be performed using [`py.test`](docs.pytest.org):

```bash
pip install -U pytest
cd context_matters
pytest
```

<!-- The Cython files can be processed individually using the following
```bash
python setup.py build_ext --inplace
``` 
A distribution for installation elsewhere can be generated using `python setup.py sdist`.  Copy the generated compress package to Rescomp using `scp ./dist/context_matters-0.1.tar.gz username@rescomp.well.ox.ac.uk:~/` and install, after un-tarring, using `cd context_matters; python setup.py install --user`.  
-->
