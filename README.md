# path-cov

Project for Software Testing Course at 22 spring.

In order to build a py project to get path-coverage data, also based on pycfg

project reports are included

## usage

### Control Flow Garph

`Python pycfg.py xxx.py -c/-d`
-c:print the result
-d:generate png file

### Trace

`python -m trace --count -C . somefile.py ...`

## example

/pycfg/trace
`python trace_test.py math_test.py`

/pycfg
`python pycfg.py trace/math_result.cover -d`
`python pycfg.py trace/math.py -d`

the final answer is the division of the two Totalresults.

