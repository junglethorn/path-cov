def bubbleSort(arr):
    n = len(arr)

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def plus(x, y):
    return x + y


def minus(x, y):
    return x - y


def divide(x, y):
    if y == 0:
        raise ZeroDivisionError
    return x / y


def max(x, y):
    if x < y:
        return y
    elif x > y:
        return x
    if x == y:
        return x


def min(x, y):
    if x < y:
        return x
    elif x > y:
        if x == y:
            return x
        else:
            return y
