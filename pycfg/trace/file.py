# 对生成的 cover 文件进行处理

f = open('calculator.cover', 'r')
outputFile = open('output.cover', 'w')
output = []

for line in f.readlines():
    if line[0:6] == ">>>>>>":
        if output[-1].lstrip().startswith("def"):
            del output[-1]
        continue
    output.append(line[7:])

for o in output:
    print(o)
    outputFile.write(o)

f.close()
outputFile.close()
