import os


class LineAnalyzer:
    # 检查前面6个字符是否为箭头
    def containsArrow(self, line):
        return line[0:6] == ">>>>>>"

    def isIf(self, line):
        return line.lstrip().startswith("if")

    def isFor(self, line):
        return line.lstrip().startswith("for")

    def isWhile(self, line):
        return line.lstrip().startswith("while")

    def isDef(self, line):
        return line.lstrip().startswith("def")

    def getPreviousBlankCount(self, line):
        count = 0
        for l in line:
            if l != " ":
                break
            count = count + 1
        return count


class CoverFileGenerator:
    def __init__(self, file):
        self.lines = []
        self.file = file
        self.analyzer = LineAnalyzer()
        self.stack = []
        self.shouldRepeat = True

    def gen(self, output):
        self.preProcessing()
        self.repeatProcessing()

        # 输出结果文件
        for o in self.lines:
            output.write(o)

    # 先对数字和箭头进行处理
    def preProcessing(self):
        # 以函数为单位，每个 def，压到栈上
        # 如果下个 def 来时，栈顶元素为 def，则该 def 没有被执行
        for line in self.file.readlines():
            # 如果是箭头，则意味着该代码没有被执行，于是直接跳过
            if self.analyzer.containsArrow(line):
                continue
            # 如果是 def 语句，需要检查栈的元素
            if self.analyzer.isDef(line[7:]):
                self.processDef()
                # 对栈进行初始化，使得新的 def 语句压栈
                self.stack.clear()
                # 把新的 def 语句压到栈上
                self.stack.append(line[7:])
                continue
            # 对剩下的语句，直接压栈
            self.stack.append(line[7:])

    # 反复遍历 lines，直到不需要再次去做处理
    def repeatProcessing(self):
        # 如果 lines 有改动，则 shouldRepeat 为 True，意味着需要重新遍历 lines
        while self.shouldRepeat:
            self.shouldRepeat = False
            for index, _ in enumerate(self.lines):
                self.processBlankLine(index)
                self.processNormal(index)

    def addLine(self, line):
        self.lines.append(line)

    def lastLine(self):
        if len(self.lines) == 0:
            raise IndexError
        return self.lines[-1]

    def deleteLastLine(self):
        if len(self.lines) == 0:
            raise IndexError
        del self.lines[-1]

    def hasPreviousAndNext(self, index):
        return index >= 1 and index < len(self.lines) - 1

    def isPreviousAndNextEqualTo(self, index, s):
        return self.lines[index + 1] == s and self.lines[index - 1] == s

    def isStackOnlyContainsDef(self):
        temp = self.stack[1:]
        for s in temp:
            if s != "\n":
                return False
        return True

    # 用栈，对 def 语句进行处理
    def processDef(self):
        if len(self.stack) == 0:
            return
        # 栈最低元素一定是 def 语句
        # 栈中不仅包含 def 语句，意味着该 def 语句被执行过，于是加到 lines 里
        if self.isStackOnlyContainsDef() == False:
            self.lines = self.lines + self.stack

    # 检查该行的上下行是否为空
    def processBlankLine(self, index):
        if self.lines[index] != "\n":
            return
        # 如果是，意味着该行是无用的一个空行，可以删除
        if self.hasPreviousAndNext(index) and self.isPreviousAndNextEqualTo(index, "\n"):
            del self.lines[index]
            self.shouldRepeat = True

    # 遇到 if（for，while） 语句，通过比较 indent 去检查下行的有效性
    def processNormal(self, index):
        if self.analyzer.isIf(self.lines[index]) == False:
            return
        if self.analyzer.isFor(self.lines[index]) == False:
            return
        if self.analyzer.isWhile(self.lines[index]) == False:
            return
        # 如果与下行的 indent 相等，意味着 if（for，while） 语句的执行部分是空的，于是删除
        if self.compareBlankCount(index, index + 1) == 0:
            del self.lines[index]
            self.shouldRepeat = True

    def compareBlankCount(self, index1, index2):
        return self.analyzer.getPreviousBlankCount(self.lines[index1]) - self.analyzer.getPreviousBlankCount(self.lines[index2])


def makeCoverFile(filename):
    file = open('%s' % (filename + ".cover"), 'r', encoding='UTF8')
    outputFile = open('%s_result.cover' % filename, 'w')

    g = CoverFileGenerator(file=file)
    g.gen(outputFile)

    file.close()
    outputFile.close()

    # 删除额外生成的 packages._distutils_hack.__init__.cover 文件
    for f in os.listdir("./"):
        if "site-packages" in f:
            os.remove(os.path.join("./", f))

    return True


def main(filename):
    file = open('%s' % (filename + ".cover"), 'r', encoding='UTF8')
    outputFile = open('%s_result.cover' % filename, 'w')

    g = CoverFileGenerator(file=file)
    g.gen(outputFile)

    file.close()
    outputFile.close()

    # 删除额外生成的 packages._distutils_hack.__init__.cover 文件
    for f in os.listdir("./"):
        if "site-packages" in f:
            os.remove(os.path.join("./", f))

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be analyzed')
    args = parser.parse_args()
    main(args.pythonfile)
