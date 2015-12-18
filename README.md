# Linear Chain CRF

A toy implementation in python to help me get my head around them. Based on
[video](http://videolectures.net/cikm08_elkan_llmacrf/) and
[pdf](http://cseweb.ucsd.edu/~elkan/250Bwinter2012/loglinearCRFs.pdf)
tutorials by Charles Elkan.

The main file is the `pos.inynb` notebook, which pulls in functions from `utils.py` and `crf.py`, and uses `project_imports3` from [myutils](https://github.com/d10genes/myutils).


# Dev

    export MYPYPATH=`pwd`/stubs
    ls test.py utils.py crf.py | entr sh onchange.sh
