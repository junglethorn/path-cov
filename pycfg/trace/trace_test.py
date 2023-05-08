import sys
import trace


def main(filename):
    tracer = trace.Trace(countfuncs=True)
    file = open('%s' % filename, 'r', encoding='UTF8')
    str = file.read()
    tracer.run(str)

    results = tracer.results()

    results.write_results(coverdir='./tests')
    file.close()

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be analyzed')
    args = parser.parse_args()
    main(args.pythonfile)
