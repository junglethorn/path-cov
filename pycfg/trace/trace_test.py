import sys
import trace
from cover import makeCoverFile


def main(filename):
    tracer = trace.Trace(
        ignoredirs=[sys.prefix, sys.exec_prefix],
        trace=0)
    file = open('%s' % filename, 'r', encoding='UTF8')
    str = file.read()
    tracer.run(str)

    results = tracer.results()

    results.write_results(coverdir='.')
    file.close()

    pureName = getPureName(filename)
    if makeCoverFile(pureName):
        print("cover file generated")

    return


def getPureName(inputFileName):
    lastIndex = inputFileName.find("_test.py")
    return inputFileName[0:lastIndex]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be analyzed')
    args = parser.parse_args()
    main(args.pythonfile)
