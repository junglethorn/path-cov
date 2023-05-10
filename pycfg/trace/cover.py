import os


# 对生成的 cover 文件进行处理
def makeCoverFile(filename):
    file = open('%s' % (filename + ".cover"), 'r', encoding='UTF8')
    outputFile = open('%s_result.cover' % filename, 'w')
    output = []

    for line in file.readlines():
        if line[0:6] == ">>>>>>":
            if output[-1].lstrip().startswith("def"):
                del output[-1]
            continue
        output.append(line[7:])

    for o in output:
        outputFile.write(o)

    file.close()
    outputFile.close()

    # 删除额外生成的 packages._distutils_hack.__init__.cover 文件
    for f in os.listdir("./"):
        if "site-packages" in f:
            os.remove(os.path.join("./", f))

    return True


def main(filename):
    file = open('%s' % filename, 'r', encoding='UTF8')
    outputFile = open('result.cover', 'w')
    output = []

    for line in file.readlines():
        if line[0:6] == ">>>>>>":
            if output[-1].lstrip().startswith("def"):
                del output[-1]
            continue
        output.append(line[7:])

    for o in output:
        outputFile.write(o)

    file.close()
    outputFile.close()

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be analyzed')
    args = parser.parse_args()
    main(args.pythonfile)
