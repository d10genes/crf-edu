
# Dev

    export MYPYPATH=`pwd`/stubs
    echo "utils.py" | entr sh -c 'mypy utils.py && python utils.py && echo "=> LOOKS GOOD!"'
